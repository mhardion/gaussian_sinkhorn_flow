import torch
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
from scipy.stats import norm
from geomloss import SamplesLoss
from typing import Callable

import copy

def sqnorm(x: torch.Tensor):
    return (x*x).sum(dim=-1)

def euclidean_cost(x, y):
    normx = sqnorm(x)
    normy = sqnorm(y)
    return normx[:,None] + normy[None,:] - 2*x@y.T

def matrix_sqrt(mat: torch.Tensor) -> torch.Tensor:
    eigvals, eigvecs = torch.linalg.eig(mat)
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


def plot_gaussian_flow(ax, covariances, covariance_target=None, means=None, mean_target=None, subsample=1, cmap = plt.get_cmap('viridis'), kw={}, kw_target={}):
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


def plot_gaussian_flow_1D(ax, vars, xm, xM, rotate=False, var_target = None, cmap=plt.get_cmap('viridis'), subsample=1):
    x = np.linspace(xm, xM, 100)
    colors = cmap(np.linspace(0, 1, len(vars[::subsample])))
    for v, c in zip(vars[::subsample], colors):
        if not rotate:
            if v > 1e-10:
                ax.plot(x, norm.pdf(x, scale=np.sqrt(v)), color=c)
            else:
                ax.axvline(0, color=c)
        else:
            if v > 1e-10:
                ax.plot(norm.pdf(x, scale=np.sqrt(v)), x, color=c)
            else:
                ax.axhline(0, color=c)
            ax.xaxis.set_inverted(True)
    if var_target is not None:
        if not rotate:
            if var_target > 1e-10:
                ax.plot(x, norm.pdf(x, scale=np.sqrt(var_target)), color='red')
            else:
                print("here")
                ax.axvline(0, color='red')
        else:
            if var_target > 1e-10:
                ax.plot(norm.pdf(x, scale=np.sqrt(var_target)), x, color='red')
            else:
                ax.axhline(0, color='red')


def plot_cv_speed(ax, sigma_0_list, sigma_star_list, blur_list, t, labels, cmap=plt.get_cmap('cividis'),showylabel=True, showlegend=True):
    for blur, sigma_0, sigma_star, label, color in zip(blur_list, sigma_0_list, sigma_star_list, labels, cmap(np.linspace(0, 1, len(blur_list)))):
        sigma_t = GaussianWGFS(blur, sigma_star).integrate(sigma_0, t)
        eps = blur ** 2
        normalized_losses = [1.]
        BS0 = BS(eps, sigma_t[0], sigma_star).item()
        for sigma in sigma_t[1:]:
            normalized_losses.append(BS(eps, sigma, sigma_star).item()/BS0)
        ax.plot(t, normalized_losses, label=label, color=color, linewidth=4)
    ax.set_yscale('log')
    ax.set_xlabel("$t$", fontdict=dict(size=20))
    if showylabel:
        ax.set_ylabel(r"$\frac{S_\varepsilon(\mu_t, \mu_\bigstar)}{S_\varepsilon(\mu_0, \mu_\bigstar)}$", fontdict=dict(size=30))
    ax.tick_params(axis='both', which="major", labelsize=15, length=10, width=2)
    ax.tick_params(axis='both', which="minor", length=5, width=1)
    if showlegend:
        ax.legend(fontsize=15)
    return ax

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
    f12, g12 = (
        softmin(eps, c_xy, (log_µ2 + g12 / eps).detach()),
        softmin(eps, c_yx, (log_µ1 + f12 / eps).detach()),
    )
    return f12, g12

def clampedlog(a):
    loga = torch.log(a)
    loga[a <= 0] = -1e5
    return loga

def train_model(model, target:Callable|torch.Tensor, t, blur=1, inner_steps=30,
                max_steps=150, tol=1e-3, sample_size=500, lr=1e-3, average_Seps_over=5, params_storage=None):
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    eps = blur**2

    if not callable(target):
        assert target.size(0) >= sample_size
        def target_sampler(N):
            index = torch.randperm(target.size(0))[:N]
            return target[index]
    else:
        target_sampler = target

    µ = torch.ones(sample_size, device=device)/sample_size



    Skdiv = SamplesLoss(blur=blur, backend='tensorized')

    with torch.no_grad():
        l = []
        for _ in range(average_Seps_over):
            l.append(Skdiv(target_sampler(sample_size),
                           model(torch.randn(sample_size, model.input_dim, device=device))).item())
        l = np.array(l)
        losses = [l.mean()]
        stdivs = [np.std(l)]

    if params_storage is not None:
        params_storage.append(copy.deepcopy(model.state_dict()))

    for i, tau in enumerate(t.diff()):
        model_frozen = copy.deepcopy(model).eval()
        model_frozen.requires_grad_(False)
        for _ in range(inner_steps):
            e = torch.randn(sample_size, model.input_dim, device=device)
            e_ = torch.randn(sample_size, model.input_dim, device=device)
            x = model(e)
            x_prev = model_frozen(e)
            x_prev_ = model_frozen(e_)
            y = target_sampler(sample_size)
            with torch.no_grad():
                c_xy = .5*euclidean_cost(x_prev, y)
                c_xx = .5*euclidean_cost(x_prev, x_prev_)
                f_cross, g_cross = sinkhorn_loop(µ, µ, c_xy, c_xy.T, [eps]*max_steps, tol=tol, init=(f_cross, g_cross) if i else None)
                f_self, g_self = sinkhorn_loop(µ, µ, c_xx, c_xx.T, [eps]*max_steps, tol=tol, init=(f_self, g_self) if i else None)
                K_self = torch.exp((f_self[:,None] + g_self[None, :] -c_xx)/eps)/sample_size
                K_cross = torch.exp((f_cross[:,None] + g_cross[None,:]-c_xy)/eps)/sample_size
                gradient = (K_self @ x_prev_ - K_cross @ y)
                x_drifted = x_prev - tau*gradient
            
            loss = torch.mean((x - x_drifted) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            l = []
            for _ in range(average_Seps_over):
                l.append(Skdiv(target_sampler(sample_size),
                           model(torch.randn(sample_size, model.input_dim, device=device))).item())
            l = np.array(l)
            losses.append(l.mean())
            stdivs.append(np.std(l))

        if params_storage is not None:
            params_storage.append(copy.deepcopy(model.state_dict()))
        print(f'{i+1}/{len(t)-1}', end='\r')

    return np.array(losses), np.array(stdivs)