import os
import glob
import math
import hashlib
import shutil
from pathlib import Path
import io

import nibabel as nib
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageSequence, ImageDraw, ImageFont
from cv2 import resize, INTER_NEAREST
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import ListedColormap
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from skimage.measure import label as cc_label, regionprops
from scipy.ndimage import (
    binary_fill_holes,
    binary_dilation,
    binary_closing,
)  
from skimage.morphology import disk  
import pandas as pd


os.makedirs('results/temp', exist_ok=True)
os.makedirs('results/gifs', exist_ok=True)
os.makedirs('results/masks', exist_ok=True)
os.makedirs('results/edited_sax_df', exist_ok=True)

GIF_H = GIF_W = 150
DISPLAY_H = DISPLAY_W = 500
OVERLAY_COLORS = [(0,0,0,0), (0,0,255,128), (255,0,0,128)]

preprocessed_gif_path = f'results/temp/preprocessed.gif'
edv_esv_gif_path = f'results/temp/edv_esv.gif'
edited_gif_path = f'results/temp/edited.gif'
raw_curve_path = f'results/temp/raw_metrics.png'
edited_curve_path = f'results/temp/edited_metrics.png'

# --------------------------------------------------------------
# Initialization
# --------------------------------------------------------------
def initialize_app(data_path, sax_series_uid, preprocess=True):
    
    # Store the last selected UID in session_state
    if "last_sax_uid" not in st.session_state:
        st.session_state.last_sax_uid = None

    # If user changes series UID, clear relevant session state
    if st.session_state.last_sax_uid != sax_series_uid:
        keys_to_clear = [
            "preprocessed",
            "edited_mask",
            "mask_hash",
            "edv_esv_selected",
            "slice_idx",
            "initialized_all",
            # any other series-specific keys
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.last_sax_uid = sax_series_uid

    if "initialized_all" in st.session_state:
        return

    raw_image = load_nii(f'{data_path}/image___{sax_series_uid}.nii.gz')
    raw_mask = load_nii(f'{data_path}/masks___{sax_series_uid}.nii.gz').astype('uint8')
    sax_df = pd.read_csv(f'{data_path}/saxdf___{sax_series_uid}.csv')

    pixelspacing, thickness = float(sax_df['pixelspacing'].iloc[0]), float(sax_df['thickness'].iloc[0])

    N = len(np.unique(raw_mask))
    raw_mask = np.eye(N, dtype=np.uint8)[raw_mask]
    raw_shape = raw_image.shape

    # -----------------------------
    # Compute raw indices
    # -----------------------------
    volume = np.sum(raw_mask[...,-1], axis=(0,1,2))
    raw_dia_idx = int(np.argmax(volume))
    raw_sys_idx = int(np.argmin(volume))

    # Compute raw metrics
    raw_volume, raw_masses, raw_edv, raw_esv, raw_sv, raw_ef, raw_mass = calculate_sax_metrics(
        raw_mask, pixelspacing, thickness, raw_dia_idx, raw_sys_idx
    )

    st.session_state.raw = {
        "image": raw_image,
        "mask": raw_mask,
        "shape": raw_shape,
        "raw_dia_idx": raw_dia_idx,
        "raw_sys_idx": raw_sys_idx,
        "raw_edv":raw_edv,
        "raw_esv":raw_esv,
        "raw_sv":raw_sv,
        "raw_ef":raw_ef,
        "raw_mass":raw_mass,
        "raw_volume": raw_volume,
        'pixelspacing':pixelspacing,
        'thickness':thickness
    }

    # -----------------------------
    # Initialize EDV/ESV selection
    # -----------------------------
    if "edv_esv_selected" not in st.session_state:
        st.session_state.edv_esv_selected = {"dia_idx": None, "sys_idx": None, "confirmed": False}

    # -----------------------------
    # Preprocess / crop if required
    # -----------------------------
    if preprocess:
        x_min, y_min, x_max, y_max = find_crop_box(np.max(raw_mask[...,-1], axis=(-1,-2)), crop_factor=1.5)
        preprocessed_image = raw_image[y_min:y_max, x_min:x_max, :, :]
        preprocessed_mask = raw_mask[y_min:y_max, x_min:x_max, :, :, :].astype('uint8')
        H, W, D, T, N = preprocessed_mask.shape

        make_video(preprocessed_image, preprocessed_mask, save_file=preprocessed_gif_path)

        has_masks = np.where(np.sum(preprocessed_mask[...,-1], axis = (0,1,3))>0)[0]
        mid_slice = len(has_masks)//2
        make_video(preprocessed_image[:,:,has_masks[mid_slice-3:mid_slice+3],:], preprocessed_mask[:,:,has_masks[mid_slice-3:mid_slice+3],:, :], save_file=edv_esv_gif_path)
        
        gif = Image.open(edv_esv_gif_path)

        st.session_state.preprocessed = {
            "image": preprocessed_image,
            "mask": preprocessed_mask,
            "H": H, "W": W, "D": D, "T": T, "N": N,
            "edv_esv_frames": [frame.copy() for frame in ImageSequence.Iterator(gif)],
            'crop_box':[x_min, y_min, x_max, y_max]
        }
    else:
        # No preprocessing, just use raw
        st.session_state.preprocessed = {
            "image": raw_image,
            "mask": raw_mask,
            "H": raw_shape[0], "W": raw_shape[1], "D": raw_shape[2], "T": raw_shape[3], "N": N,
            "frames": None,
            'crop_box':[0, 0, raw_image.shape[0], raw_image.shape[1]]

        }

    # -----------------------------
    # Plot raw curve
    # -----------------------------
    plot_volume_curve(
        raw_volume, raw_volume,
        raw_dia_idx, raw_sys_idx, raw_dia_idx, raw_sys_idx,
        save_path=raw_curve_path
    )


    # -----------------------------
    # Initialize edited mask
    # -----------------------------
    st.session_state.edited_mask = st.session_state.preprocessed["mask"].copy()
    st.session_state.mask_hash = mask_hash(st.session_state.preprocessed["mask"])
    st.session_state.initialized_all = True




def mask_hash(mask_array):
    return hashlib.md5(mask_array.tobytes()).hexdigest()


def load_nii(nii_path):
    file = nib.load(nii_path)
    data = file.get_fdata(caching='unchanged')
    return data


def thicken_close_and_fill(
    strokes: np.ndarray,
    dilate_iter: int = 1,
    close_radius: int = 2,
) -> np.ndarray:
    """
    Helper for LV contour editing:

    1. Thicken strokes (binary_dilation) so small gaps shrink.
    2. Binary closing (dilate+erode) to bridge breaks.
    3. Fill interior (binary_fill_holes).

    Input/Output: boolean mask.
    """
    if strokes is None or not strokes.any():
        return strokes

    # Small 3x3-ish disk to thicken the strokes
    selem_dilate = disk(1)
    # Slightly larger disk for closing, to bridge small end-point gaps
    selem_close = disk(close_radius)

    thick = binary_dilation(
        strokes,
        structure=selem_dilate,
        iterations=dilate_iter,
    )
    closed = binary_closing(
        thick,
        structure=selem_close,
    )
    filled = binary_fill_holes(closed)
    return filled


def make_video(image, mask, save_file, scale=1):
    position = image.shape[2]
    timesteps = image.shape[3]

    grid_rows = int(np.sqrt(position) + 0.5)
    grid_cols = (position + grid_rows - 1) // grid_rows

    H, W = image.shape[:2]
    H_scaled, W_scaled = round(GIF_H * scale), round(GIF_W * scale)

    img_min, img_max = np.min(image), np.max(image)



    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", int(18 * scale))
    except:
        font = ImageFont.load_default()

    frames = []

    for t in range(timesteps):
        canvas = Image.new(
            "RGBA",
            (grid_cols * W_scaled, grid_rows * H_scaled),
            color=(0, 0, 0, 255)
        )

        draw_canvas = ImageDraw.Draw(canvas)

        for idx in range(position):
            row, col = divmod(idx, grid_cols)

            img_slice = ((image[:,:,idx,t] - img_min) / (img_max - img_min + 1e-9) * 255).astype(np.uint8)
            img_rgb = np.stack([img_slice]*3, axis=-1)
            img_pil = Image.fromarray(img_rgb, mode="RGB").convert("RGBA")

            # Resize slice
            img_pil = img_pil.resize((W_scaled, H_scaled), resample=Image.NEAREST)

            overlay = np.zeros((H, W, 4), dtype=np.uint8)
            for ch in [1, 2]:
                ch_mask = mask[:,:,idx,t,ch]
                if np.any(ch_mask):
                    color = np.array(OVERLAY_COLORS[ch], dtype=np.uint8)
                    overlay[ch_mask > 0] = color
            overlay_pil = Image.fromarray(overlay, mode="RGBA").resize((W_scaled, H_scaled), resample=Image.NEAREST)
            img_pil.alpha_composite(overlay_pil)

            draw_tile = ImageDraw.Draw(img_pil)
            draw_tile.rectangle([0,0,int(28*scale), int(22*scale)], fill=(211,211,211,255))
            draw_tile.text((3*scale,2*scale), f"{idx}", fill=(0,0,0,255), font=font)

            canvas.paste(img_pil, (col * W_scaled, row * H_scaled), img_pil)

        draw_canvas.rectangle(
            [canvas.width - int(60*scale), canvas.height - int(20*scale),
             canvas.width, canvas.height],
            fill=(211,211,211,255)
        )
        draw_canvas.text(
            (canvas.width - int(55*scale), canvas.height - int(20*scale)),
            f"{t:02}/{timesteps - 1:02}",
            fill=(0,0,0,255),
            font=font
        )

        frames.append(canvas.convert("RGB"))

    imageio.mimsave(save_file, frames, fps=timesteps/2, loop=0)



def find_crop_box(mask, crop_factor):
    '''
    Calculated a bounding box that contains the masks inside.

    Parameters:
    mask: np.array
        A binary mask array, which should be the flattened 3D multislice mask, where the pixels in the z-dimension are summed
    crop_factor: float
        A scaling factor for the bounding box
    Returns:
    list
        A list containing the coordinates of the bounding box [x_min, y_min, x_max, y_max]. These co-ordinates can be used to crop each slice of the input multislice image.
    '''
    # Check shape of the input is 2D
    if len(mask.shape) != 2:
        raise ValueError("Input mask must be a 2D array")
    
    y = np.sum(mask, axis=1) # sum the masks across columns of array, returns a 1D array of row totals
    x = np.sum(mask, axis=0) # sum the masks across rows of array, returns a 1D array of column totals

    top = np.min(np.nonzero(y)) - 1 # Returns the indices of the elements in 1d row totals array that are non-zero, then finds the minimum value and subtracts 1 (i.e. top extent of mask)
    bottom = np.max(np.nonzero(y)) + 1 # Returns the indices of the elements in 1d row totals array that are non-zero, then finds the maximum value and adds 1 (i.e. bottom extent of mask)

    left = np.min(np.nonzero(x)) - 1 # Returns the indices of the elements in 1d column totals array that are non-zero, then finds the minimum value and subtracts 1 (i.e. left extent of mask)
    right = np.max(np.nonzero(x)) + 1 # Returns the indices of the elements in 1d column totals array that are non-zero, then finds the maximum value and adds 1 (i.e. right extent of mask)
    if abs(right - left) > abs(top - bottom):
        largest_side = abs(right - left) # Find the largest side of the bounding box
    else:
        largest_side = abs(top - bottom)
    x_mid = round((left + right) / 2) # Find the mid-point of the x-length of mask
    y_mid = round((top + bottom) / 2) # Find the mid-point of the y-length of mask
    half_largest_side = round(largest_side * crop_factor / 2) # Find half the largest side of the bounding box (crop factor scales the largest side to ensure whole heart and some surrounding is captured)
    x_max, x_min = round(x_mid + half_largest_side), round(x_mid - half_largest_side) # Find the maximum and minimum x-values of the bounding box
    y_max, y_min = round(y_mid + half_largest_side), round(y_mid - half_largest_side) # Find the maximum and minimum y-values of the bounding box
    if x_min < 0:
        x_max -= x_min # if x_min less than zero, expand the x_max value by the absolute value of x_min, to ensure bounding box is same size
        x_min = 0

    if y_min < 0:
        y_max -= y_min # if y_min less than zero, expand the y_max value by the absolute value of y_min, to ensure bounding box is same size
        y_min = 0

    return [x_min, y_min, x_max, y_max]






def calculate_sax_metrics(mask, pixelspacing, thickness, dia_idx, sys_idx):
    voxel_size = pixelspacing ** 2 * thickness / 1000
    volume = np.sum(mask[..., -1], axis=(0,1,2)) * voxel_size
    masses = np.sum(mask[..., -2], axis=(0,1,2)) * voxel_size * 1.05
    mass = masses[dia_idx]
    edv = volume[dia_idx]
    esv = volume[sys_idx]
    sv = edv - esv
    ef = (sv) * 100/edv
    return volume, masses, edv, esv, sv, ef, mass


def _label_vline(ax, x, color, y_pad=0.02):
    y0, y1 = ax.get_ylim()
    y = y0 + (y1 - y0) * y_pad
    ax.text(
        x + 0.5,
        y,
        f"{x}",
        color=color,
        fontsize=10,
        ha="center",
        va="bottom",
        rotation=90,
        alpha = 0.75
    )


def plot_volume_mass_curve(
    raw_volume,
    raw_masses,
    edited_volume,
    edited_masses,
    raw_dia_idx,
    raw_sys_idx,
    dia_idx,
    sys_idx,
    save_path,
):
    
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.25), sharex=True)

    frames_raw = np.arange(len(raw_volume))
    frames_edit = np.arange(len(edited_volume))

    edv = edited_volume[dia_idx]
    esv = edited_volume[sys_idx]
    dia_mass = edited_masses[dia_idx]

    raw_color = "#CBCBCB"
    vol_color = "#f66161"
    mass_color = "#499bed"

    axes[0].plot(frames_raw, raw_volume, color=raw_color, linewidth=2, alpha=0.7)
    axes[0].plot(
        frames_edit,
        edited_volume,
        color=vol_color,
        linewidth=2,
        label=f"EDV: {edv:.1f} mL | ESV: {esv:.1f} mL",
    )
    axes[0].set_xticks(np.arange(len(edited_volume)))


    axes[0].axvline(raw_dia_idx, color=raw_color, linestyle="--", linewidth=1.5, alpha=0.75)
    axes[0].axvline(raw_sys_idx, color=raw_color, linestyle=":", linewidth=1.5, alpha=0.75)
    axes[0].axvline(dia_idx, color=vol_color, linestyle="--", linewidth=1.5, alpha=0.75)
    axes[0].axvline(sys_idx, color=vol_color, linestyle=":", linewidth=1.5, alpha=0.75)

    _label_vline(axes[0], raw_dia_idx, raw_color)
    _label_vline(axes[0], raw_sys_idx, raw_color)
    _label_vline(axes[0], dia_idx, vol_color)
    _label_vline(axes[0], sys_idx, vol_color)

    axes[0].set_ylabel("Volume (mL)")
    axes[0].set_xlim(0, len(edited_volume) - 1)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1), edgecolor="none")

    axes[1].plot(frames_raw, raw_masses, color=raw_color, linewidth=2, alpha=0.7)
    axes[1].plot(
        frames_edit,
        edited_masses,
        color=mass_color,
        linewidth=2,
        label=f"Mass: {dia_mass:.1f} g",
    )

    axes[1].axvline(raw_dia_idx, color=raw_color, linestyle="--", linewidth=1.5, alpha=0.75)
    axes[1].axvline(dia_idx, color=mass_color, linestyle="--", linewidth=1.5, alpha=0.75)
    axes[1].set_xticks(np.arange(len(edited_volume)))

    _label_vline(axes[1], raw_dia_idx, raw_color)
    _label_vline(axes[1], dia_idx, mass_color)

    axes[1].set_xlabel("Frames")
    axes[1].set_ylabel("Mass (g)")
    axes[1].set_xlim(0, len(edited_volume) - 1)
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1), edgecolor="none")

    plt.subplots_adjust(hspace=0.05, top=1, bottom=0)
    plt.savefig(save_path, bbox_inches="tight", dpi = 400)
    plt.close(fig)

def plot_volume_curve(
    raw_volume,
    edited_volume,
    raw_dia_idx,
    raw_sys_idx,
    dia_idx,
    sys_idx,
    save_path,
):

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    frames_raw = np.arange(len(raw_volume))
    frames_edit = np.arange(len(edited_volume))

    edv = edited_volume[dia_idx]
    esv = edited_volume[sys_idx]

    raw_color = "#CBCBCB"
    vol_color = "#f66161"

    ax.plot(frames_raw, raw_volume, color=raw_color, linewidth=2, alpha=0.7)
    ax.plot(
        frames_edit,
        edited_volume,
        color=vol_color,
        linewidth=2,
        label=f"EDV: {edv:.1f} mL | ESV: {esv:.1f} mL",
    )

    ax.axvline(raw_dia_idx, color=raw_color, linestyle="--", linewidth=1.5, alpha=0.75)
    ax.axvline(raw_sys_idx, color=raw_color, linestyle=":", linewidth=1.5, alpha=0.75)
    ax.axvline(dia_idx, color=vol_color, linestyle="--", linewidth=1.5, alpha=0.75)
    ax.axvline(sys_idx, color=vol_color, linestyle=":", linewidth=1.5, alpha=0.75)

    _label_vline(ax, raw_dia_idx, raw_color)
    _label_vline(ax, raw_sys_idx, raw_color)
    _label_vline(ax, dia_idx, vol_color)
    _label_vline(ax, sys_idx, vol_color)

    ax.set_xlabel("Frames")
    ax.set_ylabel("Volume (mL)")
    ax.set_xticks(np.arange(len(edited_volume)))
    ax.set_xlim(0, len(edited_volume) - 1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1), edgecolor="none")

    plt.savefig(save_path, bbox_inches="tight", dpi=400)
    plt.close(fig)





def confirm_selection(dia_idx, sys_idx):
    """Store confirmed EDV/ESV indices in session state."""
    st.session_state.edv_esv_selected.update({
        "dia_idx": dia_idx,
        "sys_idx": sys_idx,
        "confirmed": True
    })


def frame_index_slider(
    T,
    frames,
    initial_idx,
    label,
    disabled_flag,
):
    idx = st.slider(
        f"{label} | *{initial_idx}*",
        0,
        T - 1,
        value=initial_idx,
        disabled=disabled_flag
    )
    st.image(frames[idx], use_container_width=True)
    return idx


def edv_esv_view(raw_curve_path, frames, raw_dia_idx, raw_sys_idx, T, edited_gif_path):
    """Full EDV/ESV Finder view layout."""
    if "edv_esv_selected" not in st.session_state:
        st.session_state.edv_esv_selected = {"dia_idx": None, "sys_idx": None, "confirmed": False}

    disabled_flag = st.session_state.edv_esv_selected["confirmed"]

    _, col_center,_ = st.columns([0.2,0.6,0.2])
    with col_center:
        col_edv, _, col_esv = st.columns([0.45,0.1,0.45])

        with col_edv:
            dia_idx = frame_index_slider(T, frames, raw_dia_idx, 'EDV Index', disabled_flag)

        with col_esv:
            sys_idx = frame_index_slider(T, frames, raw_sys_idx, 'EDV Index',disabled_flag)

        st.write('')
        if not disabled_flag:
            st.button(
                "Confirm EDV | ESV",
                on_click=lambda: confirm_selection(dia_idx, sys_idx),
                type="primary",
                use_container_width=True
            )
        else:
            st.success("EDV | ESV Confirmed!")

    # Generate video if confirmed
    if st.session_state.edv_esv_selected["confirmed"]:
        dia_idx = st.session_state.edv_esv_selected["dia_idx"]
        sys_idx = st.session_state.edv_esv_selected["sys_idx"]
        make_video(
            st.session_state.preprocessed["image"][:, :, :, [dia_idx, sys_idx]],
            st.session_state.preprocessed["mask"][:, :, :, [dia_idx, sys_idx], :],
            save_file=edited_gif_path
        )


def slice_navigation(D):
    """Slice navigation UI with slider and Previous/Next buttons."""
    if "slice_idx" not in st.session_state:
        st.session_state.slice_idx = 0

    def prev_slice(): st.session_state.slice_idx = max(0, st.session_state.slice_idx - 1)
    def next_slice(): st.session_state.slice_idx = min(D-1, st.session_state.slice_idx + 1)

    st.session_state.slice_idx = st.slider("Slice Index", 0, D-1, st.session_state.slice_idx, key="slice_slider")
    col_prev, col_next = st.columns(2)
    with col_prev:
        st.button("Previous", on_click=prev_slice, use_container_width=True)
    with col_next:
        st.button("Next", on_click=next_slice, use_container_width=True)

    return st.session_state.slice_idx



def update_mask_from_canvas(canvas_result, mask_state, d, idx, channel, action, H, W, N):
    if canvas_result is None or canvas_result.image_data is None:
        return mask_state
    objects = canvas_result.json_data.get("objects", [])
    if not objects:
        return mask_state

    canvas_img = canvas_result.image_data.astype(np.uint8)
    drawn_resized = np.array(Image.fromarray(canvas_img).resize((W, H), resample=Image.NEAREST))
    mask_bin = np.any(drawn_resized[:, :, :3] != 0, axis=-1).astype(np.uint8)
    mask_bin = thicken_close_and_fill(mask_bin)

    if action == "Erase ✂️":
        mask_state[:, :, d, idx, channel][mask_bin > 0] = 0
    else:
        for i in range(1, N):
            if i != channel:
                mask_state[:, :, d, idx, i][mask_bin > 0] = 0
        mask_state[:, :, d, idx, channel][mask_bin > 0] = 1

    return mask_state

def get_overlay(img_slice, mask_state, H, W, N, OVERLAY_COLORS):
    overlay = Image.fromarray(np.stack([img_slice]*3, axis=-1)).convert("RGBA")
    for i in range(1, N):
        ch_mask = mask_state[:, :, i]
        if np.any(ch_mask):
            mask_img = np.zeros((H, W, 4), dtype=np.uint8)
            mask_img[ch_mask > 0] = OVERLAY_COLORS[i]
            overlay = Image.alpha_composite(overlay, Image.fromarray(mask_img))
    overlay = overlay.resize((DISPLAY_H, DISPLAY_W), resample=Image.NEAREST)
    return overlay

def select_brush(N):
    """Brush selection UI for channel, action, and stroke width."""
    st.caption('Brush Stroke Selection')
    channel_labels = ["Myocardium 🔵", "Blood Pool 🔴"] + [f"Channel {i}" for i in range(3, N)]
    channel = st.pills("Mask", options=list(range(1, N)),
                       format_func=lambda x: channel_labels[x-1],
                       selection_mode="single", default=1)
    action = st.pills("Action", options=["Paint ✏️", "Erase ✂️"],
                      selection_mode="single", default="Paint ✏️")
    stroke_width_map = {"very fine":5,"fine":10,"medium":20,"thick":30,"very thick":60}
    stroke_width_sel = st.pills("Stroke width", options=list(stroke_width_map.keys()),
                                selection_mode="single", default=list(stroke_width_map.keys())[2])
    return channel, action, stroke_width_map[stroke_width_sel]

def draw_segmentation(img_slice, mask_slice, H, W, N, OVERLAY_COLORS, stroke_width, stroke_color, action, d, idx, channel):
    """Draw on a slice using st_canvas and update edited mask."""
    overlay = get_overlay(img_slice, mask_slice, H, W, N, OVERLAY_COLORS)
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

    if canvas_result and canvas_result.image_data is not None:
        objects = canvas_result.json_data.get("objects", [])
        if objects:
            canvas_img = canvas_result.image_data.astype(np.uint8)
            drawn_resized = np.array(Image.fromarray(canvas_img).resize((W,H), resample=Image.NEAREST))
            mask_bin = np.any(drawn_resized[:,:,:3] != 0, axis=-1).astype(np.uint8)
            mask_bin = thicken_close_and_fill(mask_bin)
            edited_mask = st.session_state.edited_mask
            if action == "Erase ✂️":
                edited_mask[:,:,d,idx,channel][mask_bin > 0] = 0
            else:
                for i in range(1, N):
                    if i != channel:
                        edited_mask[:,:,d,idx,i][mask_bin > 0] = 0
                edited_mask[:,:,d,idx,channel][mask_bin > 0] = 1
            st.session_state.edited_mask = edited_mask
            st.rerun()


def display_corrected_mask(image, edited_mask, dia_idx, sys_idx, idx_label, edited_gif_path):
    """Display corrected mask as GIF or static frame."""
    st.caption('Corrected Mask')
    view_mode = st.radio('View', ['GIF','Static'], index=0, horizontal=True)

    if mask_hash(edited_mask) != st.session_state.mask_hash:
        with st.spinner('Generating Corrected Mask...'):
            make_video(
                image[:,:,:, [dia_idx, sys_idx]],
                edited_mask[:,:,:, [dia_idx, sys_idx], :],
                save_file=edited_gif_path
            )
        st.session_state.mask_hash = mask_hash(edited_mask)

    if view_mode == 'GIF':
        st.image(edited_gif_path)
    else:
        gif = Image.open(edited_gif_path)
        frames = [frame.copy() for frame in ImageSequence.Iterator(gif)]
        view_idx = 0 if idx_label == "End-Diastole" else 1
        st.image(frames[view_idx])


def mask_editor_view(N, D, H, W, image, edited_mask, dia_idx, sys_idx, OVERLAY_COLORS, edited_gif_path):
    """Full Mask Editor layout."""
    if not st.session_state.edv_esv_selected["confirmed"]:
        st.error("Select and confirm EDV/ESV first.")
        st.stop()

    col1, col2, col3 = st.columns(3)

    with col1:
        channel, action, stroke_width = select_brush(N)
        st.write('###')
        st.caption('Image Selection')
        idx_label = st.pills("Frame", options=["End-Diastole","End-Systole"], selection_mode="single", default="End-Diastole")
        d = slice_navigation(D)

    with col2:
        st.caption('Segmentation Editor')
        edit_mode = st.radio('Mode',['Editor','Viewer'], index=0, horizontal=True)
        idx = dia_idx if idx_label=="End-Diastole" else sys_idx
        img_slice = ((image[:,:,d,idx] - image[:,:,d,idx].min()) / 
                     (image[:,:,d,idx].max() - image[:,:,d,idx].min()) * 255).astype(np.uint8)
        stroke_color = "rgba(200,200,200,0.75)" if action == "Erase ✂️" else f"rgba{OVERLAY_COLORS[channel][:3]+(0.9,)}"
        if edit_mode == 'Viewer':
            st.image(img_slice, width=DISPLAY_W)
        else:
            draw_segmentation(img_slice, edited_mask[:,:,d,idx,:], H, W, N, OVERLAY_COLORS, stroke_width, stroke_color, action, d, idx, channel)

        if st.button('Clear Whole Slice', type='primary'):
            edited_mask[:,:,d,idx,0] = 1
            edited_mask[:,:,d,idx,1:] = 0
            st.session_state.edited_mask = edited_mask
            st.rerun()

    with col3:
        display_corrected_mask(image, edited_mask, dia_idx, sys_idx, idx_label, edited_gif_path)
