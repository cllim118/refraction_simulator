import numpy as np
from matplotlib.colors import LinearSegmentedColormap

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

    s_values = np.linspace(1, 1.6, 120)
    errors = []

    for s in s_values:

        proj_x = dx0 * s + cx
        proj_y = dy0 * s + cy

        err = np.sqrt(np.nansum((proj_x - u_grid)**2 + (proj_y - v_grid)**2))
        errors.append(err)

    errors = np.array(errors)
    best_s = s_values[np.nanargmin(errors)]
    best_error = errors[np.nanargmin(errors)]

    return s_values, errors, best_s, best_error