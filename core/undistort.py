import numpy as np
from core.optics import matrix_K, trace_rays
from core.utilities_optimise import get_inair_world, trace_underwater


def jacobian_pwater_vec(u, v, depth_at_uv, kwargs, h=0.5):
    Pu1 = trace_underwater(u + h, v, Z0=depth_at_uv, **kwargs)
    Pu0 = trace_underwater(u - h, v, Z0=depth_at_uv, **kwargs)
    Pv1 = trace_underwater(u, v + h, Z0=depth_at_uv, **kwargs)
    Pv0 = trace_underwater(u, v - h, Z0=depth_at_uv, **kwargs)

    dPx_du = (Pu1[..., 0] - Pu0[..., 0]) / (2*h)
    dPy_du = (Pu1[..., 1] - Pu0[..., 1]) / (2*h)
    dPx_dv = (Pv1[..., 0] - Pv0[..., 0]) / (2*h)
    dPy_dv = (Pv1[..., 1] - Pv0[..., 1]) / (2*h)

    return np.stack([np.stack([dPx_du, dPx_dv], axis=-1),
                      np.stack([dPy_du, dPy_dv], axis=-1)], axis=-2)


def newton_solve_vec(P_air, u0, v0, depth_map, kwargs, W, H,
                      max_iter=20, tol=1e-9, damping=1.0):
    shape  = u0.shape
    u_flat = u0.astype(np.float64).ravel().copy()
    v_flat = v0.astype(np.float64).ravel().copy()
    P_air_flat = P_air.reshape(-1, P_air.shape[-1])
    depth_flat = depth_map.astype(np.float64).ravel()
    active     = np.ones(u_flat.shape, dtype=bool)

    for it in range(max_iter):
        if not np.any(active):
            break
        idx = np.where(active)[0]
        u_act, v_act = u_flat[idx], v_flat[idx]
        P_air_act    = P_air_flat[idx]
        depth_act    = depth_flat[idx]

        P_water = trace_underwater(u_act, v_act, Z0=depth_act, **kwargs)
        r         = P_water[..., :2] - P_air_act[..., :2]
        converged = np.sum(r*r, axis=-1) < tol
        active[idx[converged]] = False
        still = ~converged
        if not np.any(still):
            break

        u_act, v_act = u_act[still], v_act[still]
        depth_act    = depth_act[still]
        r            = r[still]
        idx_active   = idx[still]

        J   = jacobian_pwater_vec(u_act, v_act, depth_act, kwargs)
        det = J[..., 0, 0]*J[..., 1, 1] - J[..., 0, 1]*J[..., 1, 0]
        det_safe = np.where(np.abs(det) < 1e-12, np.nan, det)
        inv00, inv01 =  J[..., 1, 1]/det_safe, -J[..., 0, 1]/det_safe
        inv10, inv11 = -J[..., 1, 0]/det_safe,  J[..., 0, 0]/det_safe

        delta_u = inv00*r[..., 0] + inv01*r[..., 1]
        delta_v = inv10*r[..., 0] + inv11*r[..., 1]
        singular = np.isnan(delta_u) | np.isnan(delta_v)
        delta_u  = np.where(singular, 0.0, delta_u)
        delta_v  = np.where(singular, 0.0, delta_v)

        u_act = np.clip(u_act - damping*delta_u, -W, 2*W)
        v_act = np.clip(v_act - damping*delta_v, -H, 2*H)
        u_flat[idx_active] = u_act
        v_flat[idx_active] = v_act
        active[idx_active[singular]] = False

    return u_flat.reshape(shape), v_flat.reshape(shape)


def find_best_s(map_x, map_y, u_grid, v_grid, cx, cy):
    xn = map_x - cx
    yn = map_y - cy
    xd = u_grid - cx
    yd = v_grid - cy
    return np.sum(xn*xd + yn*yd) / np.sum(xn*xn + yn*yn)


def build_undistort_map(fx, fy, cx, cy, depth_map, kwargs, W, H,
                         K, n_port, rflat, tglass, mu_a, mu_g, mu_w):
    """
    depth_map: (H,W) per-pixel depth, 또는 단일 float면 전체에 동일 Z0로 broadcast
    """
    u_grid, v_grid = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))

    if np.isscalar(depth_map):
        depth_map = np.full((H, W), depth_map, dtype=np.float64)

    Z_ref = float(np.median(depth_map))
    map_x, map_y, _ = trace_rays(H, W, K, kwargs['K_inv'], n_port, rflat, tglass, Z_ref, mu_a, mu_g, mu_w)
    best_s = find_best_s(map_x, map_y, u_grid, v_grid, cx, cy)
    best_s = 1.4

    fx_s, fy_s = fx * best_s, fy * best_s
    P_air = get_inair_world(u_grid, v_grid, fx_s, fy_s, cx, cy, depth_map)

    map_u, map_v = newton_solve_vec(P_air, u_grid, v_grid, depth_map, kwargs, W, H)
    return map_u.astype(np.float32), map_v.astype(np.float32), best_s, Z_ref