import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import minimize_scalar

blue_gray = LinearSegmentedColormap.from_list('blue_gray', ['black', (0.80, 0.92, 1.0)])

def make_checkerboard(P_world, board_cols, board_rows, square_size, background=127):
    X = P_world[..., 0]
    Y = P_world[..., 1]

    board_w = board_cols * square_size
    board_h = board_rows * square_size
    x0      = -board_w / 2.0
    y0      = -board_h / 2.0
    X_local = X - x0
    Y_local = Y - y0

    mask    = (X_local >= 0) & (X_local < board_w) & (Y_local >= 0) & (Y_local < board_h)
    squares = np.zeros_like(X, dtype=np.int32) + background

    ix = np.floor(X_local[mask] / square_size).astype(int)
    iy = np.floor(Y_local[mask] / square_size).astype(int)
    squares[mask] = np.where((ix + iy) % 2 == 0, 255, 0)

    return squares


def compute_disparity(map_x, map_y, u_grid, v_grid):
    dx = map_x - u_grid
    dy = map_y - v_grid
    return np.sqrt(dx**2 + dy**2)


def sweep_s_gt(map_x, map_y, u_grid, v_grid, cx, cy):
    dx0 = map_x - cx
    dy0 = map_y - cy

    def cost(s):
        proj_x    = dx0 * s + cx
        proj_y    = dy0 * s + cy
        per_pixel = np.sqrt((proj_x - u_grid)**2 + (proj_y - v_grid)**2)
        return np.sqrt(np.nanmean(per_pixel**2))  # RMSE

    result = minimize_scalar(cost, bounds=(1.3, 1.55), method='bounded')

    best_s    = result.x
    best_rmse = result.fun

    # best_s에서 추가 지표
    proj_x    = dx0 * best_s + cx
    proj_y    = dy0 * best_s + cy
    per_pixel = np.sqrt((proj_x - u_grid)**2 + (proj_y - v_grid)**2)
    best_count = np.nansum(per_pixel < 1.0)
    sse        = np.nansum(per_pixel**2)

    # 시각화용 s_values, count_under_px
    s_values       = np.linspace(1.3, 1.55, 80)
    count_under_px = []
    for s in s_values:
        px = np.sqrt(((dx0*s+cx) - u_grid)**2 + ((dy0*s+cy) - v_grid)**2)
        count_under_px.append(np.nansum(px < 1.0))
    count_under_px = np.array(count_under_px)

    return s_values, count_under_px, best_s, best_count, best_rmse