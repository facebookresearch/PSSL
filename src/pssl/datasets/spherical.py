from scipy.special import sph_harm
import torch


# Spherical helper functions
def spherical_harmonic(x, num, return_eigenvalues=False, real=True, **kwargs):
    out = torch.zeros((len(x), num), **kwargs)
    phi = torch.arccos(x[:, 2])
    theta = torch.arctan2(x[:, 1], x[:, 0])
    i = 0
    eigenvalues = torch.zeros(num, **kwargs)
    for i in range(num):
        deg = int(i ** (1 / 2))
        freq = i - deg**2 - deg
        if real:
            out[:, i] = torch.real(sph_harm(freq, deg, theta, phi))
        else:
            out[:, i] = sph_harm(freq, deg, theta, phi)
        eigenvalues[i] = deg * (deg + 1)

    if return_eigenvalues:
        return out, eigenvalues
    return out


def spherical_eigenvalues(k):
    L = torch.ceil(torch.Tensor([k ** (1 / 2)])).int()
    eigenvalues = torch.zeros(L**2)
    for degree in range(L):
        eigenvalues[degree ** 2:(degree + 1) ** 2] = degree * (degree + 1)
    eigenvalues = eigenvalues[:k]
    return eigenvalues


def meshgrid_3d(n, **kwargs):
    theta = torch.linspace(0, 2 * torch.pi, n, **kwargs)
    phi = torch.linspace(0, torch.pi, n, **kwargs)
    X = torch.outer(torch.cos(theta), torch.sin(phi))
    Y = torch.outer(torch.sin(theta), torch.sin(phi))
    Z = torch.outer(torch.ones(n, **kwargs), torch.cos(phi))
    return X, Y, Z
