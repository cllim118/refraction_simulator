import numpy as np
from scipy.interpolate import griddata
from core.optics import normalize_batch, refract_batch, intersect_plane_batch


def compute_housing_geometry(H, W, K_inv, n_port, rflat, tglass, mu_a, mu_g, mu_w):

    u_grid, v_grid = np.meshgrid(np.arange(W, dtype=np.float32),
                                  np.arange(H, dtype=np.float32))
    pixels = np.stack([u_grid, v_grid, np.ones((H, W), dtype=np.float32)], axis=-1)
    rays   = normalize_batch(pixels @ K_inv.T)

    plane1    = n_port * rflat
    P1        = intersect_plane_batch(np.zeros(3), rays, plane1, n_port)
    ray_glass = refract_batch(rays, n_port, mu_a, mu_g)

    plane2    = plane1 + n_port * tglass
    P2        = intersect_plane_batch(P1, ray_glass, plane2, n_port)
    ray_water = refract_batch(ray_glass, n_port, mu_g, mu_w)

    return P2, ray_water


def build_forward_map_depth_term(P2, ray_water, depth_map, fx, fy, cx, cy):

    Z = depth_map.astype(np.float32)

    denom = ray_water[..., 2]
    safe_denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)

    t = (Z - P2[..., 2]) / safe_denom
    P_water_x = P2[..., 0] + t * ray_water[..., 0]
    P_water_y = P2[..., 1] + t * ray_water[..., 1]

    map_x = (fx * P_water_x / Z + cx).astype(np.float32)
    map_y = (fy * P_water_y / Z + cy).astype(np.float32)

    return map_x, map_y


def make_inverse_map(map_x, map_y, H, W, step=4):
    """forward map → inverse map (griddata)"""
    u_vals = np.arange(0, W, step)
    v_vals = np.arange(0, H, step)
    u_grid_s, v_grid_s = np.meshgrid(u_vals, v_vals)

    map_x_s = map_x[np.ix_(v_vals, u_vals)]
    map_y_s = map_y[np.ix_(v_vals, u_vals)]

    u_grid_full, v_grid_full = np.meshgrid(np.arange(W, dtype=np.float32),
                                            np.arange(H, dtype=np.float32))

    src   = np.stack([map_x_s.flatten(), map_y_s.flatten()], axis=1)
    dst_x = u_grid_s.flatten().astype(np.float32)
    dst_y = v_grid_s.flatten().astype(np.float32)

    undist_x = griddata(src, dst_x, (u_grid_full, v_grid_full), method='linear').astype(np.float32)
    undist_y = griddata(src, dst_y, (u_grid_full, v_grid_full), method='linear').astype(np.float32)
    return undist_x, undist_y


def build_undistort_map_depthterm(fx, fy, cx, cy, depth_map, P2, ray_water, W, H, best_s=1.4):
    if np.isscalar(depth_map):
        depth_map = np.full((H, W), depth_map, dtype=np.float32)

    map_x, map_y = build_forward_map_depth_term(P2, ray_water, depth_map, fx, fy, cx, cy)

    scaled_map_x = (map_x - cx) * best_s + cx
    scaled_map_y = (map_y - cy) * best_s + cy

    undist_x, undist_y = make_inverse_map(scaled_map_x, scaled_map_y, H, W)
    return undist_x, undist_y