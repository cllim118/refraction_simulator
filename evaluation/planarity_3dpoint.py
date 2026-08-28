import numpy as np
import matplotlib.pyplot as plt


POINTS3D = (
    "/home/chelim/chelim/colmap/output/p1_s02_v2/k_fixed/"
    "points3D.txt"
)


def load_points(path):
    return np.loadtxt(path, comments="#", usecols=(1, 2, 3))


def voxel_downsample(points, size):
    idx = np.floor(points / size).astype(np.int64)
    _, keep = np.unique(idx, axis=0, return_index=True)
    return points[keep]


def percentile_filter(points, p=1):
    lo, hi = np.percentile(points, [p, 100 - p], axis=0)
    return points[np.all((points >= lo) & (points <= hi), axis=1)]


def plane_axes(n):
    n = n / np.linalg.norm(n)
    ref = np.array([1., 0., 0.]) if abs(n[0]) < .9 else np.array([0., 1., 0.])
    u = np.cross(n, ref)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    v /= np.linalg.norm(v)
    return u, v


def analyze(points):
    center = points.mean(0)

    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    normal = vh[-1] / np.linalg.norm(vh[-1])

    dist = (points - center) @ normal
    proj = points - np.outer(dist, normal)

    u_axis, v_axis = plane_axes(normal)
    u = (proj - center) @ u_axis
    v = (proj - center) @ v_axis

    D = np.hypot(np.ptp(u), np.ptp(v))

    radius = np.linalg.norm(proj - center, axis=1)
    valid = radius > .01 * np.percentile(radius, 99)

    angle = np.degrees(np.arctan2(dist[valid], radius[valid]))

    return {
        "points": points,
        "center": center,
        "normal": normal,
        "dist": dist,
        "u": u,
        "v": v,
        "angle": angle,
        "valid": valid,
        "D": D,
    }


def print_results(r):
    dist, angle, D = r["dist"], r["angle"], r["D"]
    rmse = np.sqrt(np.mean(dist ** 2))
    p1, p99 = np.percentile(dist, [1, 99])

    print("\n" + "=" * 60)
    print("PLANARITY ANALYSIS")
    print("=" * 60)
    print(f"Points:          {len(dist):,}")
    print(f"Diagonal:        {D:.4f}")
    print(f"RMSE:            {rmse:.6f}")
    print(f"Relative RMSE:   {rmse / D * 100:.4f}%")
    print(f"Relative PTV:    {(p99 - p1) / D * 100:.4f}%")
    print(f"Angular RMSE:    {np.sqrt(np.mean(angle ** 2)):.4f} deg")
    print(f"Angular |P99|:   {np.percentile(np.abs(angle), 99):.4f} deg")


def plot_3d(r, max_points=10000):
    points = r["points"]

    if len(points) > max_points:
        points = points[
            np.random.default_rng(0).choice(len(points), max_points, False)
        ]

    n, c = r["normal"], r["center"]
    u, v = plane_axes(n)

    uv = np.column_stack([
        (r["points"] - c) @ u,
        (r["points"] - c) @ v
    ])

    x = np.linspace(*np.percentile(uv[:, 0], [1, 99]), 20)
    y = np.linspace(*np.percentile(uv[:, 1], [1, 99]), 20)
    xx, yy = np.meshgrid(x, y)
    plane = c + xx[..., None] * u + yy[..., None] * v

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(*points.T, s=2, alpha=.5)
    ax.plot_surface(*plane.transpose(2, 0, 1), alpha=.2)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(25, -60)

    plt.tight_layout()
    plt.show()


def plot_residual_map(r, angular=False):
    D = r["D"]

    if angular:
        valid = r["valid"]
        u = r["u"][valid] / D
        v = r["v"][valid] / D
        value = r["angle"]
        label = "Deviation (deg)"
    else:
        u = r["u"] / D
        v = r["v"] / D
        value = r["dist"] / D
        label = r"Normalized deviation $d/D$"

    c = np.percentile(np.abs(value), 99)

    fig, ax = plt.subplots(figsize=(6, 5))

    sc = ax.scatter(
        u, v,
        c=np.clip(value, -c, c),
        cmap="coolwarm",
        s=3,
        edgecolors="none"
    )

    ax.scatter(0, 0, marker="x", c="k", s=45, linewidths=1.5)

    ax.set(
        xlabel=r"$u/D$",
        ylabel=r"$v/D$",
        aspect="equal",
        xlim=(-.55, .55),
        ylim=(-.55, .55)
    )

    fig.colorbar(sc, ax=ax, pad=.02, label=label)
    plt.tight_layout()
    plt.show()
    
def plot_histogram(angle, fixed_range=None):
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.hist(angle, bins=60, range=fixed_range)
    ax.axvline(0, ls="--", lw=1.5)

    ax.set_xlabel("Deviation (deg)")
    ax.set_ylabel("Number of points")
    ax.grid(alpha=.2)

    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================

points = load_points(POINTS3D)
print(f"Loaded: {len(points):,} points")

# Downsample + remove extreme outliers
scale = np.linalg.norm(points.max(0) - points.min(0))
points = voxel_downsample(points, scale * .001)
points = percentile_filter(points)

print(f"After filtering: {len(points):,} points")

result = analyze(points)

print_results(result)

plot_3d(result)
plot_residual_map(result)
plot_residual_map(result, angular=True)
plot_histogram(result["angle"])

# plot_histogram(result["angle"], (-10, 10))