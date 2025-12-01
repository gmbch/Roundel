
from roundel_utils import *
import plotly.graph_objects as go
from scipy.ndimage import zoom
from streamlit_plotly_events import plotly_events


data_path = '/workspaces/local/Roundel/data/major_revisions'
sax_series_uid_list = sorted([uid.replace('image___','').split('/')[-1].replace('.nii.gz','') for uid in glob.glob(f'{data_path}/*') if 'image' in uid])

sax_series_uid = sax_series_uid_list[0]
# raw_mask = load_nii(f'{data_path}/masks___{sax_series_uid}.nii.gz').astype('uint8')[...,0]
# raw_mask = zoom(raw_mask, (1,1,8), order = 0)
# raw_mask = np.eye(3, dtype=np.uint8)[raw_mask]

# # --- Prepare Channels ---
# channel_1 = np.transpose(raw_mask[..., 1], (2,1,0))
# channel_2 = np.transpose(raw_mask[..., 2], (2,1,0))

# # Downsample for 3D viewer
# ds = max(1, int(max(channel_1.shape)/64))
# ch1_ds = channel_1[::ds, ::ds, ::ds]
# ch2_ds = channel_2[::ds, ::ds, ::ds]

# z1, y1, x1 = np.mgrid[0:ch1_ds.shape[0], 0:ch1_ds.shape[1], 0:ch1_ds.shape[2]]
# z2, y2, x2 = np.mgrid[0:ch2_ds.shape[0], 0:ch2_ds.shape[1], 0:ch2_ds.shape[2]]

# # --- Streamlit UI ---
# fig = go.Figure()
# fig.add_trace(go.Isosurface(x=x1.flatten(), y=y1.flatten(), z=z1.flatten(),
#                             value=ch1_ds.flatten(), isomin=0.5, isomax=ch1_ds.max(),
#                             opacity=0.25, surface_count=1, showscale=False, colorscale=[[0,'blue'],[1,'blue']]))
# fig.add_trace(go.Isosurface(x=x2.flatten(), y=y2.flatten(), z=z2.flatten(),
#                             value=ch2_ds.flatten(), isomin=0.5, isomax=ch2_ds.max(),
#                             opacity=0.25, surface_count=1, showscale=False, colorscale=[[0,'red'],[1,'red']]))
# fig.update_layout(scene=dict(aspectmode="data"), margin=dict(l=0,r=0,t=0,b=0))

# st.subheader("3D Viewer (rotate camera and click Update slice)")
# events = plotly_events(fig, select_event=False, hover_event=False, click_event=False, relayout_event=True, override_height=600)

# # extract camera from events
# camera = None
# if events:
#     for ev in events:
#         if "scene.camera" in ev:
#             camera = ev["scene.camera"]
#             st.session_state.camera = camera  # store camera for MPR

# # Button to trigger MPR extraction
# if st.button("Update slice from view") and "camera" in st.session_state:
#     cam = st.session_state.camera
#     eye = np.array([cam["eye"]["x"], cam["eye"]["y"], cam["eye"]["z"]])
#     center = np.array([cam["center"]["x"], cam["center"]["y"], cam["center"]["z"]])
#     up = np.array([cam["up"]["x"], cam["up"]["y"], cam["up"]["z"]])

#     # view direction
#     view_dir = center - eye
#     view_dir /= np.linalg.norm(view_dir)

#     # now you can feed view_dir into your arbitrary-angle MPR function
#     mpr_ch1 = mpr_arbitrary(channel_1, view_dir)
#     mpr_ch2 = mpr_arbitrary(channel_2, view_dir)

#     rgb = np.zeros(mpr_ch1.shape + (3,), dtype=np.uint8)
#     rgb[...,0] = (mpr_ch2*255).astype(np.uint8)
#     rgb[...,2] = (mpr_ch1*255).astype(np.uint8)
#     st.image(rgb, use_column_width=True)


# streamlit_app.py
#
# Requirements
# pip install streamlit plotly numpy scipy pillow nibabel streamlit-drawable-canvas streamlit-plotly-events
#
# Run
# streamlit run streamlit_app.py

import os
import numpy as np
import nibabel as nib
from PIL import Image, ImageSequence
from scipy.ndimage import map_coordinates
import plotly.graph_objects as go
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from streamlit_plotly_events import plotly_events

# -------------------------
# Configuration / constants
# -------------------------
DISPLAY_H = 512
DISPLAY_W = 512
MPR_OUTPUT_H = 512
MPR_OUTPUT_W = 512
OVERLAY_COLORS = {
    1: (0, 0, 255, 120),   # channel 1: blue semi transparent
    2: (255, 0, 0, 120),   # channel 2: red semi transparent
}

# -------------------------
# Utility IO
# -------------------------
def load_nii(path):
    img = nib.load(path)
    data = img.get_fdata()
    # Convert to uint8 if possible
    if data.dtype.kind == "f":
        data = np.asarray(data, dtype=np.float32)
    return data

# -------------------------
# MPR extraction functions
# -------------------------
def compute_camera_plane_from_plotly_camera(camera, volume_shape, spacing=(1.0, 1.0, 1.0), depth_offset=0.0):
    """
    camera: dict with keys 'eye', 'center', 'up', each mapping to dicts with x,y,z
    volume_shape: (Z, Y, X)
    spacing: voxel spacing in (z,y,x)
    returns: origin (x,y,z) in voxel coordinates, unit vectors u and v (x,y,z), normal (x,y,z)
    """
    eye = np.array([camera["eye"]["x"], camera["eye"]["y"], camera["eye"]["z"]], dtype=float)
    center = np.array([camera["center"]["x"], camera["center"]["y"], camera["center"]["z"]], dtype=float)
    up = np.array([camera["up"]["x"], camera["up"]["y"], camera["up"]["z"]], dtype=float)

    # view direction pointing from eye to center
    normal = center - eye
    norm = np.linalg.norm(normal)
    if norm == 0:
        normal = np.array([0.0, 0.0, 1.0])
    else:
        normal = normal / norm

    # Choose origin at volume center, then offset along normal by depth_offset (in voxels)
    # Map coordinate system: Plotly uses same axis order as our volume plotting (x,y,z)
    Z, Y, X = volume_shape
    center_vox = np.array([X / 2.0, Y / 2.0, Z / 2.0], dtype=float)

    origin = center_vox + normal * depth_offset

    # build in-plane basis
    # ensure up is not parallel to normal
    up_norm = up / (np.linalg.norm(up) + 1e-12)
    u = np.cross(normal, up_norm)
    u_norm = np.linalg.norm(u)
    if u_norm < 1e-6:
        # choose arbitrary perpendicular
        if abs(normal[0]) < abs(normal[1]):
            u = np.cross(normal, np.array([1, 0, 0], dtype=float))
        else:
            u = np.cross(normal, np.array([0, 1, 0], dtype=float))
        u_norm = np.linalg.norm(u)
    u = u / u_norm

    v = np.cross(normal, u)
    v = v / (np.linalg.norm(v) + 1e-12)

    return origin, u, v, normal

def extract_mpr(volume, origin, u, v, output_size=(MPR_OUTPUT_H, MPR_OUTPUT_W), spacing=(1.0, 1.0), order=1):
    """
    Sample the 3D scalar volume on an oblique plane defined by origin + a*u + b*v
    volume shape: (Z, Y, X)
    origin, u, v are in voxel coordinates and in x,y,z order
    spacing: (du, dv) sampling spacing in voxels per pixel
    Returns 2D array shape output_size (H, W)
    """
    H, W = output_size
    du, dv = spacing

    # grid coordinates in plane space: center the grid so origin is center of output
    xs = (np.arange(W) - (W - 1) / 2.0) * du
    ys = (np.arange(H) - (H - 1) / 2.0) * dv
    ii, jj = np.meshgrid(xs, ys, indexing="xy")  # ii across columns x, jj down rows y

    # compute 3D points: P = origin + ii * u + jj * v
    pts = origin[None, None, :] + ii[..., None] * u[None, None, :] + jj[..., None] * v[None, None, :]

    # convert to volume index order for map_coordinates which expects (z,y,x)
    x_coords = pts[..., 0]
    y_coords = pts[..., 1]
    z_coords = pts[..., 2]

    coords = np.vstack([
        z_coords.ravel(),
        y_coords.ravel(),
        x_coords.ravel()
    ])

    sampled = map_coordinates(volume, coords, order=order, mode="nearest")
    sampled = sampled.reshape((H, W))
    return sampled

def extract_mpr_mask(mask4d, origin, u, v, output_size=(MPR_OUTPUT_H, MPR_OUTPUT_W), spacing=(1.0, 1.0)):
    """
    mask4d: (Z, Y, X, C) where C is number of channels
    returns: (H, W, C) nearest neighbor sampled
    """
    channels = mask4d.shape[-1]
    sampled_channels = []
    for ch in range(channels):
        sampled = extract_mpr(mask4d[..., ch], origin, u, v, output_size=output_size, spacing=spacing, order=0)
        sampled_channels.append(sampled)
    return np.stack(sampled_channels, axis=-1).astype(np.uint8)

# -------------------------
# Scatter edited 2D mask back to 3D
# -------------------------
def scatter_mask_to_volume(mask_2d, volume_mask, origin, u, v, spacing=(1.0, 1.0), channel=1, slab_radius=0):
    """
    mask_2d: binary 2D mask HxW (1 where painted)
    volume_mask: 4D mask (Z, Y, X, C)
    origin, u, v define plane in voxel coords (x,y,z)
    spacing du,dv
    channel: index to write into volume_mask channels
    slab_radius: if >0, write into neighboring slices along normal within +/- slab_radius voxels
    """
    H, W = mask_2d.shape
    du, dv = spacing

    ys, xs = np.nonzero(mask_2d)
    if ys.size == 0:
        return

    xs_plane = (xs - (W - 1) / 2.0) * du
    ys_plane = (ys - (H - 1) / 2.0) * dv

    pts = origin[None, :] + xs_plane[:, None] * u[None, :] + ys_plane[:, None] * v[None, :]

    x = np.round(pts[:, 0]).astype(int)
    y = np.round(pts[:, 1]).astype(int)
    z = np.round(pts[:, 2]).astype(int)

    Z, Y, X = volume_mask.shape[:3]
    valid = (
        (x >= 0) & (x < X) &
        (y >= 0) & (y < Y) &
        (z >= 0) & (z < Z)
    )

    x = x[valid]
    y = y[valid]
    z = z[valid]

    if slab_radius <= 0:
        volume_mask[z, y, x, channel] = 1
    else:
        for dz in range(-slab_radius, slab_radius + 1):
            zz = z + dz
            good = (zz >= 0) & (zz < volume_mask.shape[0])
            volume_mask[zz[good], y[good], x[good], channel] = 1

def erase_mask_to_volume(mask_2d, volume_mask, origin, u, v, spacing=(1.0, 1.0), channel=1, slab_radius=0):
    # same as scatter but set to zero
    H, W = mask_2d.shape
    du, dv = spacing

    ys, xs = np.nonzero(mask_2d)
    if ys.size == 0:
        return

    xs_plane = (xs - (W - 1) / 2.0) * du
    ys_plane = (ys - (H - 1) / 2.0) * dv

    pts = origin[None, :] + xs_plane[:, None] * u[None, :] + ys_plane[:, None] * v[None, :]

    x = np.round(pts[:, 0]).astype(int)
    y = np.round(pts[:, 1]).astype(int)
    z = np.round(pts[:, 2]).astype(int)

    Z, Y, X = volume_mask.shape[:3]
    valid = (
        (x >= 0) & (x < X) &
        (y >= 0) & (y < Y) &
        (z >= 0) & (z < Z)
    )

    x = x[valid]
    y = y[valid]
    z = z[valid]

    if slab_radius <= 0:
        volume_mask[z, y, x, channel] = 0
    else:
        for dz in range(-slab_radius, slab_radius + 1):
            zz = z + dz
            good = (zz >= 0) & (zz < volume_mask.shape[0])
            volume_mask[zz[good], y[good], x[good], channel] = 0

# -------------------------
# Overlay utils
# -------------------------
def get_overlay_image(img_slice, mask_slice, display_size=(DISPLAY_H, DISPLAY_W), overlay_colors=OVERLAY_COLORS):
    """
    img_slice: 2D array (H, W) grayscale
    mask_slice: (H, W, C) uint8 channels with 0 or 1
    returns PIL RGBA image
    """
    H, W = img_slice.shape
    base = Image.fromarray(np.stack([img_slice, img_slice, img_slice], axis=-1).astype(np.uint8)).convert("RGBA")
    for ch in range(mask_slice.shape[-1]):
        if ch == 0:
            continue
        ch_mask = mask_slice[..., ch]
        if not ch_mask.any():
            continue
        color = overlay_colors.get(ch, (255, 255, 0, 120))
        mask_img = np.zeros((H, W, 4), dtype=np.uint8)
        mask_img[ch_mask > 0] = color
        base = Image.alpha_composite(base, Image.fromarray(mask_img))
    base = base.resize(display_size, resample=Image.NEAREST)
    return base

# -------------------------
# Plotly 3D viewer
# -------------------------
def make_plotly_volume_iso(volume, threshold=None, downsample=2):
    """
    Create a plotly isosurface scene. volume assumed shape (Z,Y,X)
    This is only for orientation and camera control.
    """
    Z, Y, X = volume.shape
    # downsample for performance
    sx = slice(None, None, downsample)
    vol_small = volume[::downsample, ::downsample, ::downsample]
    z, y, x = np.mgrid[:vol_small.shape[0], :vol_small.shape[1], :vol_small.shape[2]]
    values = vol_small.flatten()
    if threshold is None:
        threshold = np.percentile(values, 80)
    fig = go.Figure(data=go.Isosurface(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=values,
        isomin=threshold,
        isomax=values.max(),
        caps=dict(x_show=False, y_show=False, z_show=False),
        showscale=False,
        opacity=0.2
    ))
    fig.update_layout(scene=dict(aspectmode="data"), margin=dict(l=0, r=0, t=0, b=0))
    return fig

# -------------------------
# Main Streamlit app
# -------------------------
st.set_page_config(layout="wide")
st.title("3D MPR viewer and 2D editor")

load_button = st.sidebar.button("Load mask")

if "volume" not in st.session_state:
    st.session_state.volume = None
if "orig_mask" not in st.session_state:
    st.session_state.orig_mask = None
if "edited_mask" not in st.session_state:
    st.session_state.edited_mask = None
if "camera" not in st.session_state:
    # default camera
    st.session_state.camera = {
        "eye": {"x": 0.0, "y": -2.0, "z": 2.0},
        "center": {"x": 0.0, "y": 0.0, "z": 0.0},
        "up": {"x": 0.0, "y": 0.0, "z": 1.0},
    }
nii_mask_path = os.path.join(data_path, f"masks___{sax_series_uid}.nii.gz")
if load_button:
    if not os.path.exists(nii_mask_path):
        st.sidebar.error(f"Not found: {nii_mask_path}")
    else:
        raw_mask = load_nii(f'{data_path}/masks___{sax_series_uid}.nii.gz').astype('uint8')[...,0]
        raw_mask = zoom(raw_mask, (1,1,8), order = 0)
        raw_mask = np.eye(3, dtype=np.uint8)[raw_mask]
        st.session_state.orig_mask = raw_mask.copy()
        st.session_state.edited_mask = raw_mask.copy()
        
        # Reorder for Plotly/MPR: (Z,Y,X,C)
        volume_mask = np.transpose(raw_mask, (2,0,1,-1))  # (D,H,W,C) -> (Z,Y,X,C)
        st.session_state.edited_mask = volume_mask.copy()
        
        # For 3D viewer context, make a scalar intensity volume
        volume = volume_mask.sum(axis=-1).astype(np.float32)  # shape (Z,Y,X)
        st.session_state.volume = volume

# If not loaded yet, prompt
if st.session_state.volume is None:
    st.info("Load mask from sidebar. The app expects a mask NIfTI at the path above.")
    st.stop()


volume = st.session_state.volume  # shape (Z,Y,X)
edited_mask = st.session_state.edited_mask  # (Z,Y,X,C)
Z, Y, X = volume.shape

# Left column: 3D viewer
col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("3D Orientation")
    channel_1 = np.transpose(edited_mask[..., 1], (2,1,0))
    channel_2 = np.transpose(edited_mask[..., 2], (2,1,0))

    # Downsample for 3D viewer
    ds = max(1, int(max(channel_1.shape)/64))
    ch1_ds = channel_1[::ds, ::ds, ::ds]
    ch2_ds = channel_2[::ds, ::ds, ::ds]

    z1, y1, x1 = np.mgrid[0:ch1_ds.shape[0], 0:ch1_ds.shape[1], 0:ch1_ds.shape[2]]
    z2, y2, x2 = np.mgrid[0:ch2_ds.shape[0], 0:ch2_ds.shape[1], 0:ch2_ds.shape[2]]

    # --- Streamlit UI ---
    fig3d = go.Figure()
    fig3d.add_trace(go.Isosurface(x=x1.flatten(), y=y1.flatten(), z=z1.flatten(),
                                value=ch1_ds.flatten(), isomin=0.5, isomax=ch1_ds.max(),
                                opacity=0.25, surface_count=1, showscale=False, colorscale=[[0,'blue'],[1,'blue']]))
    fig3d.add_trace(go.Isosurface(x=x2.flatten(), y=y2.flatten(), z=z2.flatten(),
                                value=ch2_ds.flatten(), isomin=0.5, isomax=ch2_ds.max(),
                                opacity=0.25, surface_count=1, showscale=False, colorscale=[[0,'red'],[1,'red']]))
    fig3d.update_layout(scene=dict(aspectmode="data"), margin=dict(l=0,r=0,t=0,b=0))

    cam = st.session_state.camera
    fig3d.update_layout(scene_camera=cam)
        # render and capture events
    events = plotly_events(
        fig3d,
        select_event=False,
        hover_event=False,
        click_event=False,
        override_height=600,
    )

    # events may contain relayout event with key 'scene.camera' or 'scene.camera.eye'
    if events:
        for ev in events:
            # event from plotly-plotly_events can be a dict with 'relayout' or camera updates
            if isinstance(ev, dict):
                # try common keys
                if "scene.camera" in ev:
                    # some backends provide nested camera; attempt parse
                    cam = ev.get("scene.camera")
                    # cam may be dict with keys eye, center, up each with dicts
                    if cam:
                        st.session_state.camera = cam
                elif "scene.camera.eye.x" in ev:
                    # flatten form
                    try:
                        st.session_state.camera = {
                            "eye": {"x": ev.get("scene.camera.eye.x"), "y": ev.get("scene.camera.eye.y"), "z": ev.get("scene.camera.eye.z")},
                            "center": {"x": ev.get("scene.camera.center.x", 0.0), "y": ev.get("scene.camera.center.y", 0.0), "z": ev.get("scene.camera.center.z", 0.0)},
                            "up": {"x": ev.get("scene.camera.up.x", 0.0), "y": ev.get("scene.camera.up.y", 0.0), "z": ev.get("scene.camera.up.z", 1.0)},
                        }
                    except Exception:
                        pass
                else:
                    # some event dictionaries directly contain 'camera'
                    if "camera" in ev:
                        st.session_state.camera = ev["camera"]

    st.caption("Rotate the 3D view. When ready press Update slice from view in the editor panel.")

# Right column: MPR + editor
with col2:
    st.subheader("MPR and 2D Editor")

    # Controls
    st.caption("MPR settings")
    depth_offset = st.slider("Depth offset (voxels)", min_value=-int(max(Z, Y, X)), max_value=int(max(Z, Y, X)), value=0, step=1)
    slab_radius = st.slider("Slab radius (voxels) for brush write", min_value=0, max_value=5, value=0, step=1)

    # Brush UI
    st.caption("Brush settings")
    channel = st.selectbox("Channel", options=list(range(1, edited_mask.shape[-1])), index=0, format_func=lambda i: f"{i}")
    action = st.selectbox("Action", ["Paint", "Erase"], index=0)
    stroke_width_map = {"very fine": 5, "fine": 10, "medium": 20, "thick": 30, "very thick": 60}
    stroke_width = stroke_width_map[st.selectbox("Stroke width", list(stroke_width_map.keys()), index=2)]
    stroke_color = "#00FF00" if action == "Paint" else "#000000"

    # Button to update MPR from current camera
    if st.button("Update slice from view"):
        origin, u, v, normal = compute_camera_plane_from_plotly_camera(st.session_state.camera, (Z, Y, X), depth_offset=depth_offset)
        st.session_state.mpr = {"origin": origin, "u": u, "v": v, "normal": normal}
    # optionally initialize MPR to axial if not present
    if "mpr" not in st.session_state:
        # default axial slice at center
        origin = np.array([X / 2.0, Y / 2.0, Z / 2.0])
        u = np.array([1.0, 0.0, 0.0])
        v = np.array([0.0, 1.0, 0.0])
        normal = np.array([0.0, 0.0, 1.0])
        st.session_state.mpr = {"origin": origin, "u": u, "v": v, "normal": normal}

    mpr = st.session_state.mpr

    # Extract MPR image and mask
    img_slice = extract_mpr(volume, mpr["origin"], mpr["u"], mpr["v"], output_size=(MPR_OUTPUT_H, MPR_OUTPUT_W), spacing=(1.0, 1.0), order=1)
    mask_slice = extract_mpr_mask(edited_mask, mpr["origin"], mpr["u"], mpr["v"], output_size=(MPR_OUTPUT_H, MPR_OUTPUT_W), spacing=(1.0, 1.0))

    # Prepare overlay background for canvas
    overlay = get_overlay_image((img_slice - img_slice.min()) / max(1e-12, (img_slice.ptp() if img_slice.ptp() != 0 else 1)) * 255.0, mask_slice, display_size=(DISPLAY_H, DISPLAY_W), overlay_colors=OVERLAY_COLORS)

    # Display canvas and handle drawing
    st.caption("Draw on the overlay")
    canvas_result = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_image=overlay,
        update_streamlit=False,
        height=DISPLAY_H,
        width=DISPLAY_W,
        drawing_mode="freedraw",
        display_toolbar=True
    )

    # If drawing occurred, map edits back to 3D
    if canvas_result is not None and canvas_result.image_data is not None:
        objects = canvas_result.json_data.get("objects", [])
        if objects:
            canvas_img = canvas_result.image_data.astype(np.uint8)
            drawn_resized = np.array(Image.fromarray(canvas_img).resize((MPR_OUTPUT_W, MPR_OUTPUT_H), resample=Image.NEAREST))
            mask_bin = np.any(drawn_resized[:, :, :3] != 0, axis=-1).astype(np.uint8)

            # optional morphological thickening / close / fill can be applied here
            # For simplicity we use the raw stroke mask
            if action == "Erase":
                erase_mask_to_volume(mask_bin, st.session_state.edited_mask, mpr["origin"], mpr["u"], mpr["v"], spacing=(1.0, 1.0), channel=channel, slab_radius=slab_radius)
            else:
                # Paint: clear other channels at these pixels then set selected channel
                # Clearing other channels: sample those locations and zero them
                # We simply zero all other channels at the mapped voxels
                ys, xs = np.nonzero(mask_bin)
                if ys.size > 0:
                    # map 2D mask to 3D points to identify voxel indices, then operate on those voxels
                    xs_plane = (xs - (MPR_OUTPUT_W - 1) / 2.0)
                    ys_plane = (ys - (MPR_OUTPUT_H - 1) / 2.0)
                    pts = mpr["origin"][None, :] + xs_plane[:, None] * mpr["u"][None, :] + ys_plane[:, None] * mpr["v"][None, :]
                    x = np.round(pts[:, 0]).astype(int)
                    y = np.round(pts[:, 1]).astype(int)
                    z = np.round(pts[:, 2]).astype(int)
                    Zv, Yv, Xv = st.session_state.edited_mask.shape[:3]
                    valid = (
                        (x >= 0) & (x < Xv) &
                        (y >= 0) & (y < Yv) &
                        (z >= 0) & (z < Zv)
                    )
                    x = x[valid]; y = y[valid]; z = z[valid]
                    if slab_radius <= 0:
                        # zero other channels at these voxels
                        st.session_state.edited_mask[z, y, x, :] = 0
                        st.session_state.edited_mask[z, y, x, channel] = 1
                    else:
                        for dz in range(-slab_radius, slab_radius + 1):
                            zz = z + dz
                            good = (zz >= 0) & (zz < Zv)
                            st.session_state.edited_mask[zz[good], y[good], x[good], :] = 0
                            st.session_state.edited_mask[zz[good], y[good], x[good], channel] = 1

            # after writing back, update local mask_slice for display
            mask_slice = extract_mpr_mask(st.session_state.edited_mask, mpr["origin"], mpr["u"], mpr["v"], output_size=(MPR_OUTPUT_H, MPR_OUTPUT_W), spacing=(1.0, 1.0))
            st.experimental_rerun()

    # Show extracted MPR and overlay
    st.image(get_overlay_image((img_slice - img_slice.min()) / max(1e-12, (img_slice.ptp() if img_slice.ptp() != 0 else 1)) * 255.0, mask_slice, display_size=(DISPLAY_H, DISPLAY_W), overlay_colors=OVERLAY_COLORS), caption="Current MPR slice with overlay", use_column_width=True)

    # Provide save/export options
    col_save1, col_save2 = st.columns(2)
    with col_save1:
        if st.button("Export edited mask NIfTI"):
            # save edited mask back to disk; attempt to reuse affine from original file if possible
            out_path = os.path.join(data_path, f"edited_mask___{sax_series_uid}.nii.gz")
            # create NIfTI with identity affine; user can replace with real affine if available
            nif = nib.Nifti1Image(st.session_state.edited_mask.astype(np.uint8), affine=np.eye(4))
            nib.save(nif, out_path)
            st.success(f"Saved {out_path}")
    with col_save2:
        if st.button("Show 3 orthogonal slices"):
            # quick check: show axial, coronal, sagittal from center
            midz = Z // 2
            midy = Y // 2
            midx = X // 2
            axial = (volume[midz] - volume.min()) / max(1e-12, volume.ptp()) * 255.0
            coronal = (volume[:, midy, :] - volume.min()) / max(1e-12, volume.ptp()) * 255.0
            sagittal = (volume[:, :, midx] - volume.min()) / max(1e-12, volume.ptp()) * 255.0
            st.image([axial.astype(np.uint8), coronal.astype(np.uint8), sagittal.astype(np.uint8)], width=200, caption=["axial", "coronal", "sagittal"])

st.sidebar.caption("Notes")
st.sidebar.write("Rotate the 3D view, then press Update slice from view to extract an oblique slice aligned to the viewer. Draw on the 2D canvas to edit the mask. Edits are written back into the 3D mask.")
