import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Create output directory
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reconstruction_assets')
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------
def clean_axes(ax):
    """Remove axes and background for clean composition in Canva/Illustrator"""
    ax.set_axis_off()
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

def load_ply(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    pts = []
    colors = []
    in_data = False
    for line in lines:
        if in_data:
            parts = line.strip().split()
            if len(parts) >= 6:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                r, g, b = int(parts[3]), int(parts[4]), int(parts[5])
                pts.append([x, y, z])
                colors.append([r/255.0, g/255.0, b/255.0])
        if line.strip() == "end_header":
            in_data = True
    return np.array(pts), np.array(colors)

# ---------------------------------------------------------
# Panel 1: Full point cloud (real reconstruction)
# ---------------------------------------------------------
print("Generating Panel 1: Real Reconstruction Point Cloud...")
fig = plt.figure(figsize=(10, 10), facecolor='none')
ax = fig.add_subplot(111, projection='3d')

# Path to the actual reconstructed PLY file
ply_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs', 'reconstruction', 'reconstruction.ply')

if os.path.exists(ply_path):
    pts, colors = load_ply(ply_path)
    # Plot real points with their actual colors
    ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=colors, s=5, alpha=0.8)
else:
    # Never substitute random points for a figure labelled as a real reconstruction.
    raise SystemExit(f"{ply_path} not found. Run pipeline/main_pipeline.py on real images first.")

clean_axes(ax)
plt.savefig(os.path.join(output_dir, 'panel1_real_reconstruction.png'), transparent=True, dpi=300, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# Panel 2: The Zoom (Knot of Degeneracy)
# ---------------------------------------------------------
print("Generating Panel 2: Mid Zoom...")
fig = plt.figure(figsize=(8, 8), facecolor='none')
ax = fig.add_subplot(111, projection='3d')

if os.path.exists(ply_path):
    # Zoom into a specific subset of the points
    subset_idx = np.random.choice(len(pts), 50, replace=False)
    ax.scatter(pts[subset_idx,0], pts[subset_idx,1], pts[subset_idx,2], c=colors[subset_idx], s=30, alpha=0.9)
    ax.plot_trisurf(pts[subset_idx[:10],0], pts[subset_idx[:10],1], pts[subset_idx[:10],2], color='white', alpha=0.1)

clean_axes(ax)
plt.savefig(os.path.join(output_dir, 'panel2_zoom.png'), transparent=True, dpi=300, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# Panel 3: Types of Degeneracy (Inner Grid)
# ---------------------------------------------------------
print("Generating Panel 3: Degeneracy Types...")

fig = plt.figure(figsize=(12, 4), facecolor='none')

# 1. Reflections (Chirality)
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
base_pts = np.array([[0.5, 0.2, 0.1], [0.8, -0.4, 0.5], [0.2, -0.7, -0.2], [0.1, 0.5, -0.4]])
# Mirror across x=0
mirror_pts = base_pts.copy()
mirror_pts[:, 0] = -mirror_pts[:, 0]

ax1.scatter(base_pts[:,0], base_pts[:,1], base_pts[:,2], c='green', s=50)
ax1.scatter(mirror_pts[:,0], mirror_pts[:,1], mirror_pts[:,2], c='gray', s=50)
for i in range(len(base_pts)):
    ax1.plot([base_pts[i,0], mirror_pts[i,0]], [base_pts[i,1], mirror_pts[i,1]], [base_pts[i,2], mirror_pts[i,2]], 'w--', alpha=0.5)
# Glass plane
xx, yy = np.meshgrid(np.linspace(-1, 1, 5), np.linspace(-1, 1, 5))
zz = np.zeros_like(xx)
ax1.plot_surface(zz, xx, yy, color='blue', alpha=0.1)
clean_axes(ax1)
ax1.set_title("Reflections")

# 2. Focal Points
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
focal_point = np.array([0, 0, 0])
ax2.scatter(*focal_point, c='yellow', s=200, edgecolors='white', alpha=0.9)
# Rays converging
for _ in range(8):
    start = np.random.normal(0, 1, 3)
    # Line from start to focal point, overshooting a bit
    t = np.linspace(-1.5, 1, 10)
    ray = start * t[:, np.newaxis]
    ax2.plot(ray[:,0], ray[:,1], ray[:,2], color='amber' if 'amber' in plt.colormaps() else 'orange', alpha=0.6)
clean_axes(ax2)
ax2.set_title("Focal Points")

# 3. Non-rigid Objects
ax3 = fig.add_subplot(1, 3, 3, projection='3d')
# A wireframe slightly offset multiple times
t = np.linspace(0, 2*np.pi, 5)
x = np.cos(t)
y = np.sin(t)
z = np.zeros_like(t)
for i, offset in enumerate(np.linspace(0, 0.5, 4)):
    alpha = 1.0 - (i * 0.2)
    # Flex the object
    z_flex = z + np.sin(t + offset) * 0.5
    ax3.plot(x, y, z_flex, 'o-', color='#e07b16', alpha=alpha)
clean_axes(ax3)
ax3.set_title("Non-rigid Objects")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'panel3_degeneracy_types.png'), transparent=True, dpi=300, bbox_inches='tight')
plt.close()

print(f"Done! Assets saved in '{output_dir}' directory.")
