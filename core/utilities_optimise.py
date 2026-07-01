import numpy as np
from core.optics import normalize_batch, refract_batch, intersect_plane_batch, trace_rays
from core.utilities import make_checkerboard

EPS = 1e-12

# =========================================================
# Step 1: in-air pixel → world points
# =========================================================
def get_inair_world(u, v, fx, fy, cx, cy, Z0):

    X = (u - cx) / fx * Z0
    Y = (v - cy) / fy * Z0

    P_world = np.stack([X, Y, np.full_like(X, Z0)], axis=-1)

    return P_world

# =========================================================
# Step 2: underwater pixel (u,v) → P_water
# (with refraction: air → glass → water)
# =========================================================
def trace_underwater(u, v, K_inv, n_port, rflat, tglass, Z0,
                      mu_a, mu_g, mu_w):
    u, v  = np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64)
    ones  = np.ones_like(u)
    pixel = np.stack([u, v, ones], axis=-1)      

    ray       = normalize_batch(pixel @ K_inv.T)
    plane1    = n_port * rflat
    P1        = intersect_plane_batch(np.zeros(3), ray, plane1, n_port)

    ray_glass = refract_batch(ray, n_port, mu_a, mu_g)
    plane2    = plane1 + n_port * tglass
    P2        = intersect_plane_batch(P1, ray_glass, plane2, n_port)
    
    ray_water = refract_batch(ray_glass, n_port, mu_g, mu_w)

    denom      = ray_water[..., 2]
    safe_denom = np.where(np.abs(denom) < EPS, EPS, denom)
    t          = (Z0 - P2[..., 2]) / safe_denom
    P_water    = P2 + t[..., np.newaxis] * ray_water
    return P_water 


# =========================================================
# Step 3: cost = ||P_water(u,v) - P_world||²
# =========================================================
def cost(u, v, P_world, K_inv, n_port, rflat, tglass, Z0, mu_a, mu_g, mu_w):

    P_water = trace_underwater(u, v, K_inv, n_port, rflat, tglass, Z0,
                                mu_a, mu_g, mu_w)
    
    fx = 1.0 / K_inv[0, 0]  
    fy = 1.0 / K_inv[1, 1]
    dx = (P_world[0] - P_water[0]) / Z0 * fx
    dy = (P_world[1] - P_water[1]) / Z0 * fy
    return dx**2 + dy**2

# =========================================================
# Numerical gradient
# =========================================================
def numerical_gradient(u, v, P_world, K_inv, n_port,
                        rflat, tglass, Z0, mu_a, mu_g, mu_w, h=5):
    
    kwargs = dict(K_inv=K_inv, n_port=n_port, rflat=rflat, tglass=tglass,
                  Z0=Z0, mu_a=mu_a, mu_g=mu_g, mu_w=mu_w)
    
    du = (cost(u+h, v, P_world, **kwargs) -
          cost(u-h, v, P_world, **kwargs)) / (2*h)
    
    dv = (cost(u, v+h, P_world, **kwargs) -
          cost(u, v-h, P_world, **kwargs)) / (2*h)
    return np.array([du, dv])


# =========================================================
# Gradient descent:
# P_water(u,v) = P_world
# =========================================================
def gradient_descent(P_world, u0, v0, K_inv, n_port,
                     rflat, tglass, Z0, mu_a, mu_g, mu_w,
                     W, H, lr=0.5, max_iter=100, tol=1e-12):

    u, v    = float(u0), float(v0)
    kwargs  = dict(K_inv=K_inv, n_port=n_port, rflat=rflat, tglass=tglass,
                   Z0=Z0, mu_a=mu_a, mu_g=mu_g, mu_w=mu_w)
    history = {'cost': [], 'u': [], 'v': [],
               'P_water': []}

    for i in range(max_iter):
        c       = cost(u, v, P_world, **kwargs)
        P_water = trace_underwater(u, v, **kwargs)
        grad    = numerical_gradient(u, v, P_world, h=0.1, **kwargs)

        history['cost'].append(c)
        history['u'].append(u)
        history['v'].append(v)
        history['P_water'].append(P_water.copy())

        if c < tol:
            break

        u = np.clip(u - lr * grad[0], 0, W-1)
        v = np.clip(v - lr * grad[1], 0, H-1)

    return u, v, history