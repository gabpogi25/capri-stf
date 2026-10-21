"""
reconstruction/visualize.py

Point cloud visualization with per-point trust score coloring.

Colors:
  - Green  (R >= 0.7):  high confidence, algebraically certified
  - Orange (0.4-0.7):   moderate confidence, borderline conditioning
  - Red    (< 0.4):     low confidence, flagged as unstable

Outputs a .ply file readable by MeshLab, Blender, or Open3D.

Run:
    python reconstruction/visualize.py --input outputs/recon/reconstruction.ply
"""

from __future__ import annotations
import numpy as np
import argparse
from pathlib import Path

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def trust_score_to_color(trust_score: float) -> np.ndarray:
    """
    Map a trust score in [0, 1] to an RGB color (float, range [0,1]).

    Green -> Orange -> Red as trust decreases.

    Args:
        trust_score: R(p) in [0, 1].

    Returns:
        (3,) RGB array with values in [0, 1].
    """
    if trust_score >= 0.7:
        return np.array([0.0, 0.6, 0.2])   # green
    elif trust_score >= 0.4:
        return np.array([0.85, 0.5, 0.05])  # orange
    else:
        return np.array([0.75, 0.1, 0.1])   # red


def save_ply(
    points_3d: np.ndarray,
    trust_scores: np.ndarray,
    output_path: Path
) -> None:
    """
    Save a colored point cloud to a PLY file.

    Args:
        points_3d: (3, M) or (4, M) array of 3D point coordinates.
                   If (4, M) homogeneous, normalizes to affine automatically.
        trust_scores: (M,) array of R(p) values in [0, 1].
        output_path: Destination .ply file path.
    """
    assert points_3d.shape[1] == len(trust_scores), (
        f"Point count {points_3d.shape[1]} != trust score count {len(trust_scores)}"
    )

    # Convert homogeneous to affine if needed
    if points_3d.shape[0] == 4:
        w = points_3d[3, :]
        w = np.where(np.abs(w) < 1e-10, 1e-10, w)
        xyz = points_3d[:3, :] / w
    else:
        xyz = points_3d

    n_points = xyz.shape[1]
    colors = np.stack(
        [trust_score_to_color(float(t)) for t in trust_scores],
        axis=0
    )  # (M, 3)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(n_points):
            r, g, b = (colors[i] * 255).astype(np.uint8)
            f.write(
                f"{xyz[0,i]:.6f} {xyz[1,i]:.6f} {xyz[2,i]:.6f} "
                f"{r} {g} {b}\n"
            )


def visualize_open3d(
    points_3d: np.ndarray,
    trust_scores: np.ndarray,
    window_title: str = "CAPRI Reconstruction"
) -> None:
    """
    Display the point cloud interactively using Open3D.

    Args:
        points_3d: (3, M) or (4, M) array.
        trust_scores: (M,) array.
        window_title: Window label.
    """
    assert OPEN3D_AVAILABLE, (
        "open3d is not installed. Run: pip install open3d"
    )

    if points_3d.shape[0] == 4:
        w = points_3d[3, :]
        w = np.where(np.abs(w) < 1e-10, 1e-10, w)
        xyz = (points_3d[:3, :] / w).T  # (M, 3)
    else:
        xyz = points_3d.T

    colors = np.stack(
        [trust_score_to_color(float(t)) for t in trust_scores],
        axis=0
    )  # (M, 3)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries(
        [pcd],
        window_name=window_title,
        width=1200,
        height=800,
        point_show_normal=False
    )


def visualize_matplotlib(
    points_3d: np.ndarray,
    trust_scores: np.ndarray,
    title: str = "CAPRI Reconstruction"
) -> None:
    """
    Display the point cloud using matplotlib (fallback if Open3D unavailable).

    Args:
        points_3d: (3, M) or (4, M) array.
        trust_scores: (M,) array.
        title: Plot title.
    """
    assert MATPLOTLIB_AVAILABLE, "matplotlib is required for fallback visualization."

    if points_3d.shape[0] == 4:
        w = points_3d[3, :]
        w = np.where(np.abs(w) < 1e-10, 1e-10, w)
        xyz = points_3d[:3, :] / w
    else:
        xyz = points_3d

    colors = np.stack(
        [trust_score_to_color(float(t)) for t in trust_scores],
        axis=0
    )

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        xyz[0], xyz[1], xyz[2],
        c=colors, s=20, depthshade=True
    )
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    plt.show()


def visualize_reconstruction(
    points_3d: np.ndarray,
    trust_scores: np.ndarray,
    output_ply_path: Path = None,
    interactive: bool = True
) -> None:
    """
    Full visualization entry point: save PLY and optionally display interactively.

    Args:
        points_3d: (3, M) or (4, M) point array.
        trust_scores: (M,) array of R(p) values.
        output_ply_path: If provided, save to this path.
        interactive: If True, open interactive viewer.
    """
    if output_ply_path is not None:
        save_ply(points_3d, trust_scores, output_ply_path)
        print(f"Saved: {output_ply_path}")

    if interactive:
        if OPEN3D_AVAILABLE:
            visualize_open3d(points_3d, trust_scores)
        elif MATPLOTLIB_AVAILABLE:
            visualize_matplotlib(points_3d, trust_scores)
        else:
            print(
                "Neither open3d nor matplotlib is available. "
                "Install one for interactive visualization."
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize CAPRI reconstruction.")
    parser.add_argument("--input", type=str, required=True, help="Path to .ply file.")
    args = parser.parse_args()

    ply_path = Path(args.input)
    assert ply_path.exists(), f"File not found: {ply_path}"

    if OPEN3D_AVAILABLE:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(str(ply_path))
        o3d.visualization.draw_geometries(
            [pcd],
            window_name=f"CAPRI — {ply_path.name}",
            width=1200, height=800
        )
    else:
        print(f"Open3D not available. Open {ply_path} manually in MeshLab or Blender.")