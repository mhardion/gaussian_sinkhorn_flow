import torch
from geomloss import SamplesLoss
import numpy as np
import plotly.graph_objects as go
from abc import ABC, abstractmethod
from collections.abc import Callable
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt

def sqnorm(x: torch.Tensor):
    return (x*x).sum(dim=-1)

def euclidean_cost_batch(x, y):
    normx = sqnorm(x)
    normy = sqnorm(y)
    dotprod = torch.einsum('bik,bjk->bij', x, y)
    return normx[...,None] + normy[:,None,:] - 2*dotprod

def clampedlog(a):
    loga = torch.log(a)
    loga[a <= 0] = -1e5
    return loga

def euclidean_cost(x, y):
    normx = sqnorm(x)
    normy = sqnorm(y)
    return normx[:,None] + normy[None,:] - 2*x@y.T

# Code was adapted from the geomloss package, see https://github.com/jeanfeydy/geomloss 
def softmin(eps, c, f):
    return -eps * (f.view(1, -1) - c / eps).logsumexp(1).view(-1)

def sinkhorn_loop(µ1, µ2, c_xy, c_yx, eps_list, init=None, tol=1e-4):
    log_µ1 = clampedlog(µ1)
    log_µ2 = clampedlog(µ2)
    torch.autograd.set_grad_enabled(False)
    eps = eps_list[0]
    if init is None:
        f12 = softmin(eps, c_xy, log_µ2)
        g12 = softmin(eps, c_yx, log_µ1)
    else:
        f12, g12 = init
    # count = 0
    for eps in eps_list:
        f12_ = softmin(eps, c_xy, log_µ2 + g12 / eps)
        g12_ = softmin(eps, c_yx, log_µ1 + f12 / eps)
        # count+=1
        if max(abs(f12_ - f12).max().item(),
               abs(g12_ - g12).max().item()) < tol:
            break
        f12, g12 = 0.5 * (f12 + f12_), 0.5 * (g12 + g12_) 

    torch.autograd.set_grad_enabled(True)
    # print(count, end='\r')
    f12, g12 = (
        softmin(eps, c_xy, (log_µ2 + g12 / eps).detach()),
        softmin(eps, c_yx, (log_µ1 + f12 / eps).detach()),
    )
    return f12, g12

def self_transport(µ, c, eps, init=None, tol=1e-4, max_steps=10):
    logµ = clampedlog(µ)
    torch.autograd.set_grad_enabled(False)
    if init is None:
        f_µ = softmin(eps, c, logµ)
    else:
        f_µ = init
    k = 0
    d = tol + 1
    while k < max_steps and d > tol:
        k += 1
        f_µ_ = f_µ.clone()
        f_µ = .5*(f_µ + softmin(eps, c, logµ + f_µ / eps))
        d = abs(f_µ-f_µ_).max().item()
    
    torch.autograd.set_grad_enabled(True)
    f_µ = softmin(eps, c, (logµ + f_µ / eps).detach())
    return f_µ

class WGFS:
    def __init__(self, blur, x_target) -> None:
        self.x_target = x_target
        self.eps = blur**2

    def integrate(self, x_0, t):
        x = x_0.clone()
        n = x.size(0)
        µ = torch.ones(n)/n
        x_t = x_0[None,...]
        for tau in torch.diff(t):
            c_xy = euclidean_cost(x, self.x_target)
            c_xx = euclidean_cost(x, x)
            f_cross, g_cross = sinkhorn_loop(µ, µ, c_xy, c_xy.T, [self.eps]*50)
            f_self = self_transport(µ, c_xx, self.eps)
            K_self = torch.exp((f_self[:,None] + f_self[None, :] -c_xx)/self.eps)
            K_cross = torch.exp((f_cross[:,None] + g_cross[None,:]-c_xy)/self.eps)
            gradient = (K_self @ x - K_cross @ self.x_target)/n
            x -= tau*gradient
            x_t = torch.cat((x_t, x.clone()[None,...]))
        return x_t


def matrix_sqrt(mat: torch.Tensor) -> torch.Tensor:
    # Compute the matrix square root using eigen-decomposition
    eigvals, eigvecs = torch.linalg.eig(mat)
    # Clamp eigenvalues to avoid negative values due to numerical errors
    assert not (eigvals.imag.any() or eigvecs.imag.any())
    eigvals_clamped = torch.clamp(eigvals.real, min=0)
    sqrt_eigvals = torch.sqrt(eigvals_clamped)
    eigvecs = eigvecs.real
    return eigvecs @ torch.diag(sqrt_eigvals) @ eigvecs.T


class GaussianWGFS():
    def __init__(self, blur, sigma_target) -> None:
        self.eps = blur**2
        self.sigma_target = sigma_target
    
    def integrate(self, sigma_0, t):
        sigma = sigma_0.clone()
        Id = torch.eye(sigma.size(0))
        sqrt_sigma_target = matrix_sqrt(self.sigma_target)
        sigma_t = sigma_0[None,...]
        for tau in torch.diff(t):
            G = 2*(sigma @ torch.linalg.inv(matrix_sqrt(sigma@sigma+self.eps**2*Id/16) + self.eps*Id/4)
                   - sqrt_sigma_target @ torch.linalg.inv(matrix_sqrt(sqrt_sigma_target@sigma@sqrt_sigma_target+self.eps**2*Id/16) + self.eps*Id/4)@sqrt_sigma_target)
            sigma = (Id - tau*G) @ sigma @(Id - tau*G)
            sigma_t = torch.cat((sigma_t, sigma[None,...]))
        return sigma_t
    
    def integrate_1D(self, sigma_0, t):
        sigma_t = sigma_0[None]
        teps = self.eps/4
        sigma = sigma_0
        for tau in torch.diff(t):
            sigma += tau* 4*sigma*(self.sigma_target/(torch.sqrt(self.sigma_target*sigma + teps**2)+teps) - sigma/(torch.sqrt(sigma**2 + teps**2)+teps))
            sigma_t = torch.cat([sigma_t, sigma[None]])
        return sigma_t


def BS(eps, sig0, sig1):
    sqrt_sig0 = matrix_sqrt(sig0)
    D_cross = torch.sqrt(4*torch.linalg.eigvals(sqrt_sig0 @ sig1 @ sqrt_sig0).real +(eps**2/4))
    D_0 = torch.sqrt(4*torch.linalg.eigvals(sig0).real**2 +(eps**2/4))
    D_1 = torch.sqrt(4*torch.linalg.eigvals(sig1).real**2 +(eps**2/4))
    return (.5*(D_0 + D_1) - D_cross).sum() + .5*eps*torch.log(D_cross+.5*eps).sum() - .25*eps*torch.log(D_0+.5*eps).sum() - .25*eps*torch.log(D_1+.5*eps).sum()


def plot_gaussian_flow(ax, covariances, covariance_target=None, means=None, mean_target=None, subsample=1, cmap = plt.cm.get_cmap('viridis'), kw={}, kw_target={}):
    """
    Plots the evolution of 2D covariance matrices as ellipses using matplotlib.
    Args:
        covariances (torch.Tensor): N x 2 x 2 tensor of covariance matrices.
        means (torch.Tensor or np.ndarray, optional): N x 2 tensor of mean vectors.
        covariance_target (torch.Tensor or np.ndarray, optional): 2 x 2 target covariance matrix to plot.
        mean_target (torch.Tensor or np.ndarray, optional): 2-dim target mean vector.
    Returns:
        matplotlib.figure.Figure
    """
    covs = covariances[::subsample]
    N = covs.shape[0]
    t = np.linspace(0, 1, N)
    colors = cmap(t)

    if means is not None:
        ms = means.detach().cpu().numpy() if isinstance(means, torch.Tensor) else np.array(means)
        ms = ms[::subsample]
    else:
        ms = np.zeros((N, 2))

 
    
    for i in range(N):
        cov = covs[i].detach().cpu().numpy()
        mean = ms[i]
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = eigvals.argsort()[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        angle = np.degrees(np.arctan2(eigvecs[1,0], eigvecs[0,0]))
        width, height = 2 * np.sqrt(np.maximum(eigvals, 0))
        ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                         facecolor='none', edgecolor=colors[i], **kw)
        ax.add_patch(ellipse)

    if covariance_target is not None:
        if isinstance(covariance_target, torch.Tensor):
            cov = covariance_target.detach().cpu().numpy()
        else:
            cov = np.array(covariance_target)
        if mean_target is not None:
            if isinstance(mean_target, torch.Tensor):
                mean = mean_target.detach().cpu().numpy()
            else:
                mean = np.array(mean_target)
        else:
            mean = np.zeros(2)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = eigvals.argsort()[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        angle = np.degrees(np.arctan2(eigvecs[1,0], eigvecs[0,0]))
        width, height = 2 * np.sqrt(np.maximum(eigvals, 0))
        ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, facecolor='none', 
                         **(dict(edgecolor='red', linewidth=4, 
                         alpha=1.0, label='Target Gaussian') | kw_target))
        ax.add_patch(ellipse)
    return ax


def plot_gaussian_flow_marginals(ax_x, ax_y, covariances, covariance_target=None, means=None, mean_target=None, subsample=1, cmap = plt.cm.get_cmap('viridis'), kw={}, kw_target={}):
    covs = covariances[::subsample]
    N = covs.shape[0]
    t = np.linspace(0, 1, N)
    colors = cmap(t)

    if means is not None:
        ms = means.detach().cpu().numpy() if isinstance(means, torch.Tensor) else np.array(means)
        ms = ms[::subsample]
    else:
        ms = np.zeros((N, 2))

    for i in range(N):
        cov = covs[i].detach().cpu().numpy()
        mean = ms[i]
        
        # Plot x marginal
        var_x = cov[0, 0]
        if var_x < 1e-10:
            ax_x.axvline(mean[0], color=colors[i], **(dict(linewidth=2)|kw))
        else:
            std_x = np.sqrt(var_x)
            x_range = np.linspace(mean[0] - 4*std_x, mean[0] + 4*std_x, 200)
            density_x = (1 / (std_x * np.sqrt(2*np.pi))) * np.exp(-0.5*((x_range - mean[0])/std_x)**2)
            ax_x.plot(x_range, density_x, color=colors[i], **kw)
        
        # Plot y marginal
        var_y = np.maximum(cov[1, 1], 0)
        if var_y < 1e-10:
            ax_y.axhline(mean[1], color=colors[i], **(dict(linewidth=2) | kw))
        else:
            std_y = np.sqrt(var_y)
            y_range = np.linspace(mean[1] - 4*std_y, mean[1] + 4*std_y, 200)
            density_y = (1 / (std_y * np.sqrt(2*np.pi))) * np.exp(-0.5*((y_range - mean[1])/std_y)**2)
            ax_y.plot(density_y, y_range, color=colors[i], **kw)

    if covariance_target is not None:
        if isinstance(covariance_target, torch.Tensor):
            cov = covariance_target.detach().cpu().numpy()
        else:
            cov = np.array(covariance_target)
        if mean_target is not None:
            if isinstance(mean_target, torch.Tensor):
                mean = mean_target.detach().cpu().numpy()
            else:
                mean = np.array(mean_target)
        else:
            mean = np.zeros(2)
        
        var_x = np.maximum(cov[0, 0], 0)
        if var_x < 1e-10:
            ax_x.axvline(mean[0], **(dict(color='red', linewidth=2, label='Target') | kw_target))
        else:
            std_x = np.sqrt(var_x)
            x_range = np.linspace(mean[0] - 4*std_x, mean[0] + 4*std_x, 200)
            density_x = (1 / (std_x * np.sqrt(2*np.pi))) * np.exp(-0.5*((x_range - mean[0])/std_x)**2)
            ax_x.plot(x_range, density_x, **(dict(color='red', linewidth=2, label='Target') | kw_target))
        
        var_y = np.maximum(cov[1, 1], 0)
        if var_y < 1e-10:
            ax_y.axhline(mean[1], **(dict(color='red', linewidth=2, label='Target') | kw_target))
        else:
            std_y = np.sqrt(var_y)
            y_range = np.linspace(mean[1] - 4*std_y, mean[1] + 4*std_y, 200)
            density_y = (1 / (std_y * np.sqrt(2*np.pi))) * np.exp(-0.5*((y_range - mean[1])/std_y)**2)
            ax_y.plot(density_y, y_range, **(dict(color='red', linewidth=2, label='Target') | kw_target))
    
    return ax_x, ax_y