import numpy as np
import matplotlib.pyplot as plt


RC_PATH = "/home/chelim/chelim/colmap_underwater/output/p1_s02/k_fixed/points3D.txt"
OURS_PATH = "/home/chelim/chelim/colmap/output/p1_s02_v2/k_fixed/points3D.txt"


def load(path):
    return np.loadtxt(path, comments="#", usecols=(1, 2, 3))


def axes(n):
    n /= np.linalg.norm(n)
    ref = np.array([1., 0., 0.]) if abs(n[0]) < .9 else np.array([0., 1., 0.])
    u = np.cross(n, ref); u /= np.linalg.norm(u)
    v = np.cross(n, u); v /= np.linalg.norm(v)
    return u, v


def plane_diag(points):
    c = points.mean(0)
    _, _, vh = np.linalg.svd(points - c, full_matrices=False)
    n = vh[-1]
    u, v = axes(n)
    uv = np.column_stack([(points - c) @ u, (points - c) @ v])
    return np.hypot(np.ptp(uv[:, 0]), np.ptp(uv[:, 1]))


def voxel(points, size):
    idx = np.floor(points / size).astype(np.int64)
    _, keep = np.unique(idx, axis=0, return_index=True)
    return points[keep]


def percentile_filter(points, p=1):
    lo, hi = np.percentile(points, [p, 100-p], axis=0)
    return points[np.all((points >= lo) & (points <= hi), axis=1)]


def sample(points, n=100_000, seed=0):
    if len(points) <= n:
        return points
    rng = np.random.default_rng(seed)
    return points[rng.choice(len(points), n, replace=False)]


def analyze(points, ref_normal=None):
    c = points.mean(0)
    _, _, vh = np.linalg.svd(points - c, full_matrices=False)
    n = vh[-1] / np.linalg.norm(vh[-1])

    if ref_normal is not None and np.dot(n, ref_normal) < 0:
        n = -n

    d = -n @ c
    dist = points @ n + d
    proj = points - np.outer(dist, n)

    u_axis, v_axis = axes(n)
    u = (proj - c) @ u_axis
    v = (proj - c) @ v_axis
    D = np.hypot(np.ptp(u), np.ptp(v))

    r = np.linalg.norm(proj - c, axis=1)
    valid = r > .01 * np.percentile(r, 99)
    angle = np.degrees(np.arctan2(dist[valid], r[valid]))

    return u, v, dist, angle, D, n


def print_result(name, result):
    u, v, dist, angle, D, _ = result

    rmse = np.sqrt(np.mean(dist**2))
    p1, p99 = np.percentile(dist, [1, 99])

    print(f"\n{name}")
    print(f"Points:          {len(dist):,}")
    print(f"Diagonal:        {D:.4f}")
    print(f"RMSE:            {rmse:.6f}")
    print(f"Relative RMSE:   {rmse / D * 100:.4f}%")
    print(f"Relative PTV:    {(p99 - p1) / D * 100:.4f}%")
    print(f"Angular RMSE:    {np.sqrt(np.mean(angle**2)):.4f} deg")
    print(f"Angular |P99|:   {np.percentile(np.abs(angle), 99):.4f} deg")


def plot_results(results):
    data = [
        (r[0] / r[4], r[1] / r[4], r[2] / r[4])
        for r in results
    ]

    lim = max(max(np.abs(u).max(), np.abs(v).max())
              for u, v, _ in data)
    c = max(np.percentile(np.abs(res), 99) for _, _, res in data)

    fig, ax = plt.subplots(1, 2, figsize=(8, 3.6))

    for i, (a, (u, v, res)) in enumerate(zip(ax, data)):
        if i == 1:   # Ours
            u = u

        sc = a.scatter(
            u, v,
            c=np.clip(res, -c, c),
            cmap="coolwarm",
            s=4,
            edgecolors="none"
        )

        a.scatter(0, 0, marker="x", c="k", s=40)
        a.set(
            xlabel=r"$u/D$",
            ylabel=r"$v/D$",
            xlim=(-lim, lim),
            ylim=(-lim, lim),
            aspect="equal"
        )

    fig.subplots_adjust(right=0.86, wspace=0.25)

    cbar_ax = fig.add_axes([0.89, 0.15, 0.025, 0.7])
    fig.colorbar(sc, cax=cbar_ax,
                 label=r"Normalized residual $r/D$")

    plt.show()

# ============================================================
# Load
# ============================================================

rc = load(RC_PATH)
ours = load(OURS_PATH)

print(f"Raw RC:   {len(rc):,}")
print(f"Raw Ours: {len(ours):,}")


# ============================================================
# Scale alignment: RC -> Ours
# ============================================================

scale = plane_diag(ours) / plane_diag(rc)
rc *= scale

print(f"RC scale: {scale:.6e}")


# ============================================================
# Same voxel size + filtering
# ============================================================

voxel_size = plane_diag(ours) * 0.001

rc = percentile_filter(voxel(rc, voxel_size))
ours = percentile_filter(voxel(ours, voxel_size))

print(f"After voxel/filter RC:   {len(rc):,}")
print(f"After voxel/filter Ours: {len(ours):,}")


# ============================================================
# Same number of points
# ============================================================

N = min(100_000, len(rc), len(ours))

rc = sample(rc, N)
ours = sample(ours, N)

print(f"Evaluation points: {N:,}")


# ============================================================
# Analyze
# ============================================================

rc_result = analyze(rc)
ours_result = analyze(ours, ref_normal=rc_result[-1])

print_result("RC", rc_result)
print_result("Ours", ours_result)


# ============================================================
# Plot
# ============================================================

plot_results([
    rc_result,
    ours_result
])