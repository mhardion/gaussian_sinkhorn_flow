import torch
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
from scipy.stats import norm



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


def plot_gaussian_flow(ax, covariances, covariance_target=None, means=None, mean_target=None, subsample=1, cmap = plt.cm.get_cmap('viridis'), kw={}, kw_target={}):
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