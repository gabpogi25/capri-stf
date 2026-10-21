"""
dashboard.py

Streamlit front end for the 3D-reconstruction pipeline in
colmap_capri_automator.py: upload photos, run SuperGlue matching with the
chosen filter (none / RANSAC / RANSAC + CAPRI), COLMAP sparse (and optionally
dense) reconstruction, then inspect the result layer by layer.

What is shown is exactly what the pipeline wrote to the output directory:
  features.h5, matches_<mode>.h5, filter_stats.json, sparse/0, dense/meshed.ply
CAPRI removes matches before COLMAP runs, so removed matches have no 3D
position; the dashboard reports their number from filter_stats.json rather
than drawing them.

Run:  streamlit run dashboard.py
"""

import json
import shutil
import traceback
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

try:
    import pycolmap
except ImportError:
    pycolmap = None

try:
    import trimesh
except ImportError:
    trimesh = None

from colmap_capri_automator import FILTER_MODES, run_superglue_pipeline

st.set_page_config(page_title="CAPRI Reconstruction Dashboard", layout="wide")
st.title("CAPRI reconstruction dashboard")
st.markdown("SuperGlue matching → match filter → COLMAP. Each layer below is read from the output directory.")

col1, col2 = st.columns([1, 3])

with col1:
    st.header("1. Photos")
    uploaded_files = st.file_uploader("Images of one scene", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    st.header("2. Run")
    output_dir = Path(st.text_input("Output directory", value="outputs/dashboard"))
    mode = st.selectbox("Match filter", FILTER_MODES, index=FILTER_MODES.index("ransac_capri"))
    seed = int(st.number_input("Seed", value=42, step=1))
    dense = st.checkbox("Dense reconstruction + Poisson mesh (needs CUDA COLMAP)", value=False)

    if st.button("Run pipeline", type="primary"):
        if not uploaded_files or len(uploaded_files) < 2:
            st.error("Upload at least 2 images.")
        else:
            image_dir = Path("data/temp_uploads")
            shutil.rmtree(image_dir, ignore_errors=True)
            image_dir.mkdir(parents=True, exist_ok=True)
            shutil.rmtree(output_dir, ignore_errors=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            for f in uploaded_files:
                # COLMAP's argument parsing breaks on spaces and parentheses in file names
                safe_name = f.name.replace(" ", "_").replace("(", "").replace(")", "")
                (image_dir / safe_name).write_bytes(f.getbuffer())
            with st.spinner("Running SuperGlue, the match filter and COLMAP..."):
                try:
                    run_superglue_pipeline(str(image_dir), str(output_dir), mode=mode, seed=seed, dense=dense)
                    st.success("Reconstruction finished.")
                except Exception as e:  # surface any pipeline failure in the UI
                    st.error(f"Pipeline failed: {e}")
                    st.code(traceback.format_exc())

with col2:
    stats_path = output_dir / "filter_stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else None
    sparse_dir = output_dir / "sparse" / "0"
    sparse_exists = (sparse_dir / "cameras.bin").exists() or (sparse_dir / "cameras.txt").exists()
    mesh_path = output_dir / "dense" / "meshed.ply"

    if not output_dir.exists():
        st.warning("No output directory yet. Run the pipeline or point to an existing output.")
    else:
        if stats:
            st.subheader("Match filtering")
            c = st.columns(4)
            c[0].metric("Image pairs", stats.get("pairs", 0))
            c[1].metric("SuperGlue matches", stats.get("matches_in", 0))
            c[2].metric("After RANSAC", stats.get("matches_after_ransac", 0))
            c[3].metric(f"Kept ({stats.get('mode')})", stats.get("matches_out", 0))
            if stats.get("mode") == "ransac_capri":
                removed = stats.get("matches_after_ransac", 0) - stats.get("matches_out", 0)
                st.caption(f"CAPRI removed {removed} RANSAC-inlier matches before reconstruction "
                           f"({stats.get('pairs_capri_empty', 0)} pairs left empty by CAPRI were handled by the "
                           "empty-pair policy). Removed matches are never triangulated, so they have no 3D position.")
            with st.expander("filter_stats.json"):
                st.json(stats)

        layer = st.radio("Layer", ["Sparse points", "Sparse points + cameras", "Poisson mesh"], horizontal=True)
        fig = go.Figure()
        rec = None
        if sparse_exists and pycolmap is not None:
            try:
                rec = pycolmap.Reconstruction(str(sparse_dir))
            except Exception as e:  # corrupted or partial model
                st.error(f"Failed to load COLMAP model: {e}")
        elif sparse_exists:
            st.warning("Install pycolmap to view the sparse model.")

        if layer.startswith("Sparse") and rec is not None:
            xyz = np.array([p.xyz for p in rec.points3D.values()])
            rgb = np.array([p.color for p in rec.points3D.values()])
            if len(xyz):
                fig.add_trace(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers",
                                           marker=dict(size=2, color=[f"rgb({r},{g},{b})" for r, g, b in rgb]),
                                           name=f"{len(xyz)} triangulated points"))
            if layer.endswith("cameras"):
                xs, ys, zs = [], [], []
                for image in rec.images.values():
                    centre = image.projection_center()
                    look = image.cam_from_world.rotation.matrix().T @ np.array([0, 0, 0.5])
                    end = centre + look
                    xs += [centre[0], end[0], None]
                    ys += [centre[1], end[1], None]
                    zs += [centre[2], end[2], None]
                fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(width=3),
                                           name=f"{len(rec.images)} registered cameras"))
        elif layer == "Poisson mesh":
            if mesh_path.exists() and trimesh is not None:
                mesh = trimesh.load(mesh_path)
                v, f = mesh.vertices, mesh.faces
                colours = None
                if hasattr(mesh.visual, "vertex_colors") and len(mesh.visual.vertex_colors):
                    colours = [f"rgb({r},{g},{b})" for r, g, b in mesh.visual.vertex_colors[:, :3]]
                fig.add_trace(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
                                        vertexcolor=colours, color=None if colours else "lightblue",
                                        name="Poisson mesh"))
            else:
                st.warning("dense/meshed.ply not found (run with dense enabled) or trimesh is not installed.")

        fig.update_layout(scene=dict(aspectmode="data", xaxis=dict(visible=False), yaxis=dict(visible=False),
                                     zaxis=dict(visible=False)), margin=dict(l=0, r=0, b=0, t=0), height=750)
        st.plotly_chart(fig, use_container_width=True)
