# optics.py
import numpy as np

EPS = 1e-12

def normalize_batch(v: np.ndarray) -> np.ndarray:
    """v: (..., 3) → normalized along last axis"""
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(norm < EPS, v, v / norm)

def refract_batch(X, n, mu1, mu2):
    """
    X: (H, W, 3) ray directions (unit)
    n: (3,)      interface normal (unit, in the direction of ray propagation)
    Returns (H, W, 3), TIR pixels set to nan
    """
    r     = mu1 / mu2
    cos_i = np.einsum('...i,i->...', X, n)        
    term  = 1.0 - r**2 * (1.0 - cos_i**2)          

    tir   = term < 0
    term  = np.where(tir, 0.0, term)

    coeff    = np.sqrt(term) - r * cos_i
    refracted = r * X + coeff[..., np.newaxis] * n
    refracted[tir] = np.nan
    return refracted

def intersect_plane_batch(origin, ray, plane_point, normal):
    """
    origin:      (3,) or (H, W, 3)
    ray:         (H, W, 3)
    plane_point: (3,)
    normal:      (3,)
    Returns (H, W, 3)
    """
    denom = np.einsum('...i,i->...', ray, normal)           
    denom = np.where(np.abs(denom) < EPS, EPS, denom)

    # plane_point - origin → (H,W,3) when origin is (H,W,3), else (3,)
    diff = plane_point - origin                             
    t    = np.einsum('...i,i->...', diff, normal) / denom  
    return origin + t[..., np.newaxis] * ray 

def matrix_K(fx, fy, cx, cy):
    K     = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
    K_inv = np.linalg.inv(K)
    return K, K_inv

def trace_rays(H, W, K, K_inv, n_port, rflat, tglass, Z0,
                   mu_a, mu_g, mu_w):
    """
    Returns
    -------
    map_x, map_y : (H, W) float32  
    valid        : (H, W) bool     
    """
    # 1. Pixel grid → unit rays in air
    u, v = np.meshgrid(np.arange(W, dtype=float),
                       np.arange(H, dtype=float))
    pixels  = np.stack([u, v, np.ones((H, W))], axis=-1)
    ray_air = normalize_batch(pixels @ K_inv.T)   

    # 2. Air → glass interface
    plane1 = n_port * rflat
    P1     = intersect_plane_batch(np.zeros(3), ray_air, plane1, n_port)

    # 3. Refract air → glass
    ray_glass = refract_batch(ray_air, n_port, mu_a, mu_g)

    # 4. Glass → water interface
    plane2 = plane1 + n_port * tglass
    P2     = intersect_plane_batch(P1, ray_glass, plane2, n_port)

    # 5. Refract glass → water
    ray_water = refract_batch(ray_glass, n_port, mu_g, mu_w)

    # 6. Intersect scene plane Z = Z0
    denom     = ray_water[..., 2]
    safe_denom = np.where(np.abs(denom) < EPS, EPS, denom)
    t          = (Z0 - P2[..., 2]) / safe_denom
    P_world    = P2 + t[..., np.newaxis] * ray_water

    # 7. Project to image coords
    p = P_world @ K.T 
    p /= p[..., 2:3]

    map_x = p[..., 0].astype(np.float32)
    map_y = p[..., 1].astype(np.float32)

    return map_x, map_y, P_world