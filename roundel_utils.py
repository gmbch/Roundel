import os, sys
import glob
import math
import hashlib
import shutil
from pathlib import Path
import io
import json
from pathlib import Path
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
    binary_erosion,
    binary_closing,
    gaussian_filter
)
from skimage.morphology import disk
from skimage.measure import find_contours
import pandas as pd
import time
import cv2

blank_gif_path = f'results/temp/blank.gif'
full_edited_gif_path = f'results/temp/edited.gif'
preprocessed_gif_path = f'results/temp/preprocessed.gif'
edv_esv_gif_path = f'results/temp/edv_esv.gif'
edited_gif_path = f'results/temp/edited_edv_esv.gif'
raw_curve_path = f'results/temp/raw_metrics.png'
edited_curve_path = f'results/temp/edited_metrics.png'
cache_dir = 'cache'

os.makedirs('results/temp', exist_ok=True)
os.makedirs('results/gifs', exist_ok=True)
os.makedirs('results/masks', exist_ok=True)
os.makedirs('results/edited_sax_df', exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)

GIF_W = 150
DISPLAY_W = 400
BACKGROUND_COLOR = (100, 100, 0, 0)
LV_MYO_COLOR = (0, 255, 255, 50) # Blue
LV_COLOR = (255, 10, 10, 50)      # Red

background_idx = 0
lv_myo_idx = 1
lv_idx = 2

channels = [lv_myo_idx, lv_idx]

BRUSH_LABELS = {
    lv_myo_idx: 'Myocardium 🔵',
    lv_idx: 'Blood Pool 🔴',
}


OVERLAY_COLORS = {
    background_idx: BACKGROUND_COLOR,
    lv_myo_idx: LV_MYO_COLOR,
    lv_idx: LV_COLOR,
}


def save_config(config: dict, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(config, f, indent=2)


def load_config(path) -> dict:
    path = Path(path)
    with path.open("r") as f:
        return json.load(f)


def save_mask(mask, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    nib_mask = nib.Nifti1Image(mask, affine=np.eye(4), dtype='uint8')
    nib.save(nib_mask, save_path)


def skip_case(study_uid, patient, study_date):
    """UI handler for skipping a case."""
    from aws_utils import skip_case_ddb, fetch_staged_roundel_cases

    # Update DynamoDB
    skip_case_ddb(study_uid)

    st.warning(f"⏭️ Skipped case for {patient} ({study_date})")

    # Refresh staged list
    new_cases = fetch_staged_roundel_cases()
    new_cases = sorted(new_cases, key=lambda c: str(c.get("fid", "")))

    # Clear session state
    clear_keys = [
        "edited_mask", "edv_esv_selected", "preprocessed", "raw",
        "point1", "point2", "coord1", "coord2", "crop1", "crop2",
        "selected_case"
    ]
    for k in clear_keys:
        st.session_state.pop(k, None)

    # If none left → stop
    if not new_cases:
        st.sidebar.success("🎉 All Roundel cases completed!")
        st.stop()

    # Move to next case
    st.session_state["selected_case"] = new_cases[0]

    # Reset tab
    st.session_state["next_view"] = "EDV/ESV Finder 🔍"

    # Reload UI
    st.rerun()


def load_font(size):
    """
    try:
        font = load_font(int(18 * scale))
    except:
        font = ImageFont.load_default()
    """

    # Try Linux font
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except:
        pass
    # Try Windows font
    try:
        return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)
    except:
        pass
    # Fallback (non scalable)
    return ImageFont.load_default()


# --------------------------------------------------------------
# Initialization
# --------------------------------------------------------------
def initialize_app(data_path, study_uid, pixelspacing, thickness, preprocess=True):
    st.session_state['data_path'] = data_path
    st.session_state['study_uid'] = study_uid

    os.makedirs(f'{data_path}/cache', exist_ok=True)

    # Store the last selected UID in session_state
    if "last_sax_uid" not in st.session_state:
        st.session_state.last_sax_uid = None

    # If user changes series UID, clear relevant session state
    if st.session_state.last_sax_uid != study_uid:
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
        st.session_state.last_sax_uid = study_uid

    if "initialized_all" in st.session_state:
        return

    raw_image = load_nii(f'{data_path}/image___{study_uid}.nii.gz')
    raw_mask = load_nii(f'{data_path}/masks___{study_uid}.nii.gz').astype('uint8')
    # sax_df = pd.read_csv(f'{data_path}/saxdf___{study_uid}.csv')

    # pixelspacing, thickness = float(sax_df['pixelspacing'].iloc[0]), float(sax_df['thickness'].iloc[0])

    N = len(np.unique(raw_mask))
    raw_mask = np.eye(N, dtype=np.uint8)[raw_mask]
    raw_shape = raw_image.shape

    # -----------------------------
    # Compute raw indices
    # -----------------------------
    volume = np.sum(raw_mask[..., -1], axis=(0, 1, 2))
    raw_dia_idx = int(np.argmax(volume))
    raw_sys_idx = np.where(volume != 0)[0][np.argmin(volume[volume != 0])]

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
        "raw_edv": raw_edv,
        "raw_esv": raw_esv,
        "raw_sv": raw_sv,
        "raw_ef": raw_ef,
        "raw_mass": raw_mass,
        "raw_volume": raw_volume,
        'pixelspacing': pixelspacing,
        'thickness': thickness
    }

    # -----------------------------
    # Initialize EDV|ESV selection
    # -----------------------------
    if "edv_esv_selected" not in st.session_state:
        st.session_state.edv_esv_selected = {"dia_idx": None, "sys_idx": None, "confirmed": False}

    # -----------------------------
    # Preprocess / crop if required
    # -----------------------------
    x_min, y_min, x_max, y_max = find_crop_box(np.max(raw_mask[..., [lv_idx, lv_myo_idx]], axis=(-1, -2, -3)),
                                               crop_factor=1.5)

    subpixel_resolution = 500 // (y_max - y_min)
    subpixel_resolution = min(4, subpixel_resolution)
    st.session_state['subpixel_resolution'] = subpixel_resolution

    preprocessed_image = raw_image[y_min:y_max, x_min:x_max, :, :]
    preprocessed_mask = raw_mask[y_min:y_max, x_min:x_max, :, :, :].astype('uint8')
    H, W, D, T, N = preprocessed_mask.shape

    has_masks = np.where(np.sum(preprocessed_mask[..., -1], axis=(0, 1, 3)) > 0)[0]
    mid_slice = len(has_masks) // 2

    zoom = [st.session_state['subpixel_resolution'], st.session_state['subpixel_resolution'], 1, 1]

    smoothed_image = cv_zoom(preprocessed_image, zoom=zoom)

    st.session_state['cache_config_path'] = f"{cache_dir}/config___{study_uid}.json"
    st.session_state['cache_mask_path'] = f"{cache_dir}/masks___{study_uid}.nii.gz"

    if os.path.exists(st.session_state['cache_config_path']) and os.path.exists(st.session_state['cache_mask_path']):
        smoothed_mask = load_nii(st.session_state['cache_mask_path']).astype("uint8")
        cached = True
    else:
        smoothed_mask = cv_zoom_mask(
            preprocessed_mask,
            zoom=zoom + [1],
            interpolation=cv2.INTER_NEAREST,
        )
        cached = False

    make_video(smoothed_image[:, :, has_masks[mid_slice - 3:mid_slice + 3], :],
               smoothed_mask[:, :, has_masks[mid_slice - 3:mid_slice + 3], :, :] * 0, save_file=edv_esv_gif_path)
    make_video(smoothed_image, smoothed_mask * 0, save_file=blank_gif_path)

    gif = Image.open(edv_esv_gif_path)

    st.session_state.preprocessed = {
        "image": preprocessed_image,
        "mask": preprocessed_mask,
        "smooth_image": smoothed_image,
        "smooth_mask": smoothed_mask,
        "H": H,
        "W": W,
        "D": D,
        "T": T,
        "N": N,
        "edv_esv_frames": [frame.copy() for frame in ImageSequence.Iterator(gif)],
        "crop_box": [x_min, y_min, x_max, y_max],
    }

    if cached:
        config = load_config(st.session_state['cache_config_path'])
        confirm_selection(dia_idx=config['dia_idx'], sys_idx=config['sys_idx'])

    # -----------------------------
    # Initialize edited mask
    # -----------------------------
    st.session_state['edited_mask'] = st.session_state.preprocessed["smooth_mask"].copy()
    save_mask(st.session_state['edited_mask'], save_path=st.session_state['cache_mask_path'])
    st.session_state['mask_hash'] = mask_hash(st.session_state.preprocessed["mask"])
    st.session_state["brush_mode"] = "Paint ✏️"
    st.session_state["stroke_width"] = "thin"
    st.session_state["edit_made"] = False
    st.session_state['edited_frames'] = None
    st.session_state['cached'] = cached

    st.session_state.initialized_all = True


def cv_zoom(images, zoom, interpolation=cv2.INTER_CUBIC):
    """
    Resize height and width of a 4D or 5D array using OpenCV. Only H and W are scaled.

    Args:
        images (numpy.ndarray): Array of shape (H, W, D, T) or (H, W, D, T, C)
        zoom_factors (list or tuple): Zoom factors for (H, W, D, T, C). Only H and W > 1
        interpolation (int): OpenCV interpolation method (default: cv2.INTER_CUBIC)

    Returns:
        numpy.ndarray: Resized array with height and width scaled, other dimensions unchanged
    """
    h_zoom, w_zoom = zoom[0], zoom[1]

    if images.ndim == 4:
        h, w, d, t = images.shape
        resized = np.zeros((int(h * h_zoom), int(w * w_zoom), d, t), dtype=images.dtype)
        for z in range(d):
            for tau in range(t):
                resized[..., z, tau] = cv2.resize(images[..., z, tau], (int(w * w_zoom), int(h * h_zoom)),
                                                  interpolation=interpolation)
    elif images.ndim == 5:
        h, w, d, t, c = images.shape
        resized = np.zeros((int(h * h_zoom), int(w * w_zoom), d, t, c), dtype=images.dtype)
        for z in range(d):
            for tau in range(t):
                for ch in range(c):
                    resized[..., z, tau, ch] = cv2.resize(images[..., z, tau, ch], (int(w * w_zoom), int(h * h_zoom)),
                                                          interpolation=interpolation)
    else:
        raise ValueError("Input must be 4D or 5D array.")

    return resized


def cv_zoom_mask(
        mask,
        zoom,
        sigma=2.0,
        interpolation=cv2.INTER_CUBIC,
):
    """
    mask: H,W,D,T,C
    returns: H,W,D,T,C one-hot
    """

    zoomed = cv_zoom(mask.astype(np.float32), zoom, interpolation=interpolation)

    myo = (zoomed[..., lv_myo_idx] > 0.5).astype(np.float32)
    endo = (zoomed[..., lv_idx] > 0.5).astype(np.float32)

    epi = np.zeros_like(myo, dtype=bool)
    for d in range(myo.shape[2]):
        for t in range(myo.shape[3]):
            epi[..., d, t] = binary_fill_holes(myo[..., d, t].astype(np.uint8))

    epi = gaussian_filter(epi.astype(np.float32), sigma=(sigma, sigma, 0, 0)) > 0.5
    endo = gaussian_filter(endo.astype(np.float32), sigma=(sigma, sigma, 0, 0)) > 0.5

    # Encode labels: 0=bg, 1=endo, 2=myo
    labels = np.zeros(epi.shape, dtype=np.uint8)
    labels[epi] = lv_myo_idx
    labels[endo] = lv_idx

    # One-hot
    return np.eye(3, dtype=np.uint8)[labels]


def mask_hash(mask_array):
    return hashlib.md5(mask_array.tobytes()).hexdigest()


def load_nii(nii_path):
    file = nib.load(nii_path)
    data = file.get_fdata(caching='unchanged')
    return data


def thicken_close_fill_and_smooth(strokes, stroke_width):
    if strokes is None or not strokes.any():
        return strokes

    # Use power-law scaling for dilation
    dilation_factor = max(1, int(10 / (stroke_width ** 2)))

    # Detect contours to check for nested shapes
    dilated = binary_dilation(strokes, iterations=dilation_factor)
    contours = find_contours(dilated, 0.5)

    has_ring = False
    for i, c1 in enumerate(contours):
        for j, c2 in enumerate(contours):
            if i == j:
                continue
            y1, x1 = c1[:, 0], c1[:, 1]
            y2, x2 = c2[:, 0], c2[:, 1]
            if (y2.min() > y1.min() and y2.max() < y1.max() and
                    x2.min() > x1.min() and x2.max() < x1.max()):
                has_ring = True
                break
        if has_ring:
            break

    if has_ring:
        # Dilation + fill + erosion
        closed = binary_dilation(strokes, iterations=dilation_factor)
        filled = binary_fill_holes(closed)
        filled = binary_erosion(filled, iterations=dilation_factor)

        # Apply minor Gaussian blur and re-threshold to smooth edges
        # blurred = gaussian_filter(filled.astype(float), sigma=0.5)
        # smoothed = blurred > 0.48  # Convert back to binary
        return filled.astype('uint8')
    else:
        # For strokes without rings, apply very mild smoothing
        # blurred = gaussian_filter(strokes.astype(float), sigma=0.5)
        # smoothed = blurred > 0.48
        return strokes.astype('uint8')


def make_video(image, mask, save_file, mask_frames='all', scale=1):
    save_file = Path(save_file)
    save_file.parent.mkdir(parents=True, exist_ok=True)

    position = image.shape[2]
    timesteps = image.shape[3]

    grid_rows = int(np.sqrt(position) + 0.5)
    grid_cols = (position + grid_rows - 1) // grid_rows

    H, W = image.shape[:2]
    GIF_H = H * GIF_W / W
    H_scaled, W_scaled = round(GIF_H * scale), round(GIF_W * scale)
    img_min, img_max = np.min(image), np.max(image)

    try:
        font = load_font(int(18 * scale))
    except:
        font = ImageFont.load_default()

    frames = []
    if mask_frames == 'all':
        mask_frames = np.arange(timesteps)

    for t in mask_frames:
        canvas = Image.new(
            "RGBA",
            (grid_cols * W_scaled, grid_rows * H_scaled),
            color=(0, 0, 0, 255)
        )

        draw_canvas = ImageDraw.Draw(canvas)

        for idx in range(position):
            row, col = divmod(idx, grid_cols)

            image_slice = ((image[:, :, idx, t] - img_min) / (img_max - img_min + 1e-9) * 255).astype(np.uint8)
            img_rgb = np.stack([image_slice] * 3, axis=-1)
            img_pil = Image.fromarray(img_rgb, mode="RGB").convert("RGBA")

            # Resize slice
            img_pil = img_pil.resize((W_scaled, H_scaled), resample=Image.NEAREST)

            overlay = np.zeros((H, W, 4), dtype=np.uint8)
            if t in mask_frames:
                for ch in channels:
                    ch_mask = mask[:, :, idx, t, ch]
                    if np.any(ch_mask):
                        color = np.array(OVERLAY_COLORS[ch], dtype=np.uint8)
                        overlay[ch_mask > 0] = color
            overlay_pil = Image.fromarray(overlay, mode="RGBA").resize((W_scaled, H_scaled), resample=Image.NEAREST)
            img_pil.alpha_composite(overlay_pil)

            draw_tile = ImageDraw.Draw(img_pil)
            draw_tile.rectangle([0, 0, int(28 * scale), int(22 * scale)], fill=(211, 211, 211, 255))
            draw_tile.text((3 * scale, 2 * scale), f"{idx}", fill=(0, 0, 0, 255), font=font)

            canvas.paste(img_pil, (col * W_scaled, row * H_scaled), img_pil)

        draw_canvas.rectangle(
            [canvas.width - int(60 * scale), canvas.height - int(20 * scale),
             canvas.width, canvas.height],
            fill=(211, 211, 211, 255)
        )
        draw_canvas.text(
            (canvas.width - int(55 * scale), canvas.height - int(20 * scale)),
            f"{t:02}/{timesteps - 1:02}",
            fill=(0, 0, 0, 255),
            font=font
        )

        frames.append(canvas.convert("RGB"))

    if len(mask_frames) < 5:
        fps = len(mask_frames) / 2
    else:
        fps = np.clip(len(mask_frames) / 2, 8, 15)
    imageio.mimsave(save_file, frames, fps=fps, loop=0)


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
    # Check shape of the input is 2D
    if len(mask.shape) != 2:
        raise ValueError("Input mask must be a 2D array")

    y = np.sum(mask, axis=1)  # sum the masks across columns of array, returns a 1D array of row totals
    x = np.sum(mask, axis=0)  # sum the masks across rows of array, returns a 1D array of column totals

    top = np.min(np.nonzero(
        y)) - 1  # Returns the indices of the elements in 1d row totals array that are non-zero, then finds the minimum value and subtracts 1 (i.e. top extent of mask)
    bottom = np.max(np.nonzero(
        y)) + 1  # Returns the indices of the elements in 1d row totals array that are non-zero, then finds the maximum value and adds 1 (i.e. bottom extent of mask)

    left = np.min(np.nonzero(
        x)) - 1  # Returns the indices of the elements in 1d column totals array that are non-zero, then finds the minimum value and subtracts 1 (i.e. left extent of mask)
    right = np.max(np.nonzero(
        x)) + 1  # Returns the indices of the elements in 1d column totals array that are non-zero, then finds the maximum value and adds 1 (i.e. right extent of mask)
    if abs(right - left) > abs(top - bottom):
        largest_side = abs(right - left)  # Find the largest side of the bounding box
    else:
        largest_side = abs(top - bottom)
    x_mid = round((left + right) / 2)  # Find the mid-point of the x-length of mask
    y_mid = round((top + bottom) / 2)  # Find the mid-point of the y-length of mask
    half_largest_side = round(
        largest_side * crop_factor / 2)  # Find half the largest side of the bounding box (crop factor scales the largest side to ensure whole heart and some surrounding is captured)
    x_max, x_min = round(x_mid + half_largest_side), round(
        x_mid - half_largest_side)  # Find the maximum and minimum x-values of the bounding box
    y_max, y_min = round(y_mid + half_largest_side), round(
        y_mid - half_largest_side)  # Find the maximum and minimum y-values of the bounding box
    if x_min < 0:
        x_max -= x_min  # if x_min less than zero, expand the x_max value by the absolute value of x_min, to ensure bounding box is same size
        x_min = 0

    if y_min < 0:
        y_max -= y_min  # if y_min less than zero, expand the y_max value by the absolute value of y_min, to ensure bounding box is same size
        y_min = 0

    return [x_min, y_min, x_max, y_max]


def calculate_sax_metrics(mask, pixelspacing, thickness, dia_idx, sys_idx):
    voxel_size = pixelspacing ** 2 * thickness / 1000
    volume = np.sum(mask[..., lv_idx], axis=(0, 1, 2)) * voxel_size
    masses = np.sum(mask[..., lv_myo_idx], axis=(0, 1, 2)) * voxel_size * 1.05
    mass = masses[dia_idx]
    edv = volume[dia_idx]
    esv = volume[sys_idx]
    sv = edv - esv
    ef = (sv) * 100 / edv
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
        alpha=0.75
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
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
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
    plt.savefig(save_path, bbox_inches="tight", dpi=400)
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
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
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


def wrap(key, min_val, max_val):
    if st.session_state[key] > max_val:
        st.session_state[key] = min_val
    elif st.session_state[key] < min_val:
        st.session_state[key] = max_val


def frame_index_slider(
        T,
        frames,
        initial_idx,
        label,
        disabled_flag,
        key
):
    idx = st.slider(
        f"{label} | *{initial_idx}*",
        -1,
        T,
        value=initial_idx,
        key=key,
        on_change=wrap,
        args=(key, 0, T - 1),
        disabled=disabled_flag
    )
    st.image(frames[idx], use_container_width=True)
    return idx


def confirm_selection(dia_idx, sys_idx):
    """Store confirmed EDV|ESV indices in session state."""
    st.session_state.edv_esv_selected.update({
        "dia_idx": dia_idx,
        "sys_idx": sys_idx,
        "confirmed": True
    })

    save_config(st.session_state.edv_esv_selected, st.session_state['cache_config_path'])

    make_video(
        st.session_state.preprocessed["smooth_image"],
        st.session_state.preprocessed["smooth_mask"],
        save_file=full_edited_gif_path,
        mask_frames=[dia_idx, sys_idx]
    )
    gif = Image.open(full_edited_gif_path)
    frames = [f.copy() for f in ImageSequence.Iterator(gif)]
    st.session_state['edited_frames'] = frames


def edv_esv_view():
    """Full EDV|ESV Finder view layout."""
    if "edv_esv_selected" not in st.session_state:
        st.session_state.edv_esv_selected = {"dia_idx": None, "sys_idx": None, "confirmed": False}

    frames = st.session_state.preprocessed['edv_esv_frames']

    if st.session_state.edv_esv_selected['confirmed']:
        display_dia_idx = st.session_state.edv_esv_selected['dia_idx']
        display_sys_idx = st.session_state.edv_esv_selected['sys_idx']

    else:
        display_dia_idx = st.session_state.raw['raw_dia_idx']
        display_sys_idx = st.session_state.raw['raw_sys_idx']
    H, W, D, T, N = [st.session_state.preprocessed[k] for k in ["H", "W", "D", "T", "N"]]

    disabled_flag = st.session_state.edv_esv_selected["confirmed"]

    _, col_center, _ = st.columns([0.05, 0.9, 0.05])
    with col_center:
        col_edv, _, col_esv = st.columns([0.45, 0.1, 0.45])

        with col_edv:
            dia_idx = frame_index_slider(T, frames, display_dia_idx, 'EDV Index', disabled_flag, key='edv')

        with col_esv:
            sys_idx = frame_index_slider(T, frames, display_sys_idx, 'ESV Index', disabled_flag, key='esv')

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


def slice_navigation(D):
    if "slice_idx" not in st.session_state:
        st.session_state.slice_idx = 0
    if "previous_slice_idx" not in st.session_state:
        st.session_state.previous_slice_idx = st.session_state.slice_idx

    # Store previous slice
    previous_d = st.session_state.previous_slice_idx

    # Slider (updates slice_idx immediately)
    st.slider(
        "Slice Index",
        0,
        D - 1,
        key="slice_idx",
    )

    col_prev, col_next = st.columns(2)
    with col_prev:
        st.button(
            "Previous",
            on_click=lambda: st.session_state.update(
                slice_idx=max(0, st.session_state.slice_idx - 1)
            ),
            use_container_width=True,
        )
    with col_next:
        st.button(
            "Next",
            on_click=lambda: st.session_state.update(
                slice_idx=min(D - 1, st.session_state.slice_idx + 1)
            ),
            use_container_width=True,
        )

    # Determine if canvas needs reset
    previous_objects = st.session_state.get('canvas', {}).get('previous_objects', [])
    reset_canvas = previous_d != st.session_state.slice_idx and bool(previous_objects)

    # Update previous slice for next rerun
    st.session_state.previous_slice_idx = st.session_state.slice_idx

    return st.session_state.slice_idx, reset_canvas


def get_overlay(image_slice, mask_state, H, W, N, OVERLAY_COLORS):
    overlay = Image.fromarray(np.stack([image_slice] * 3, axis=-1)).convert("RGBA")
    for i in channels:
        ch_mask = mask_state[:, :, i]
        if np.any(ch_mask):
            mask_img = np.zeros(
                (H * st.session_state['subpixel_resolution'], W * st.session_state['subpixel_resolution'], 4),
                dtype=np.uint8)
            mask_img[ch_mask > 0] = OVERLAY_COLORS[i]
            overlay = Image.alpha_composite(overlay, Image.fromarray(mask_img))
    return overlay


def select_brush(N):
    """Brush selection UI for channel, action, and stroke width."""
    action = st.radio("Brush Stroke Selection",
                      options=["Paint ✏️", "Erase ✂️"],
                      index=["Paint ✏️", "Erase ✂️"].index(st.session_state.brush_mode),
                      horizontal=True)

    st.session_state['brush_mode'] = action
    stroke_width_map = {"thin": 6, "medium": 20, "thick": 40}

    stroke_width_sel = st.radio("Stroke Width",
                                options=list(stroke_width_map.keys()),
                                index=list(stroke_width_map.keys()).index(st.session_state["stroke_width"]),
                                horizontal=True)

    st.session_state['stroke_width'] = stroke_width_sel

    if action == "Paint ✏️":
        valid_channels = [i for i in range(N) if i != background_idx]
        channel = st.radio(
            "Mask",
            options=valid_channels,
            format_func=lambda x: BRUSH_LABELS[x],
            index=0,
            horizontal=True
        )

    else:
        channel = 0
    stroke_width = stroke_width_map[stroke_width_sel]
    return channel, action, stroke_width


def normalize(image):
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    return image


def mask_editor_view():
    """Efficient Mask Editor with controlled reruns and canvas caching."""
    if not st.session_state.edv_esv_selected["confirmed"]:
        st.error("Select and confirm EDV/ESV first.")
        st.stop()

    H, W, D, T, N = [st.session_state.preprocessed[k] for k in ["H", "W", "D", "T", "N"]]
    image = st.session_state.preprocessed["smooth_image"]
    edited_mask = st.session_state['edited_mask']
    dia_idx = st.session_state.edv_esv_selected["dia_idx"]
    sys_idx = st.session_state.edv_esv_selected["sys_idx"]

    col1, col2, col3 = st.columns([1, 1.5, 1.5])

    with col1:
        channel, action, stroke_width = select_brush(N)
        st.divider()
        idx_label = st.radio("Frame", ["End-Diastole", "End-Systole"], index=0, horizontal=True)
        d, reset_canvas = slice_navigation(D)

    idx = dia_idx if idx_label == "End-Diastole" else sys_idx

    # Normalize slice once per display
    img_slice = image[:, :, d, idx]
    image_slice = ((img_slice - img_slice.min()) / (img_slice.max() - img_slice.min()) * 255).astype(np.uint8)
    mask_slice = edited_mask[:, :, d, idx, :]

    with col2:
        edit_mode = st.radio('Segmentation Editor', ['Editor', 'Viewer'], index=0, horizontal=True)
        stroke_color = f"rgba{OVERLAY_COLORS[background_idx][:3] + (0.8,)}" if action == "Erase ✂️" else f"rgba{OVERLAY_COLORS[channel][:3] + (0.65,)}"

        if edit_mode == 'Viewer':
            st.image(image_slice, width=DISPLAY_W)
        else:
            # Initialize canvas state
            if 'canvas' not in st.session_state:
                st.session_state['canvas'] = {
                    'canvas_key': f'editor_{d}',
                    'previous_d': d,
                    'previous_objects': []
                }

            if reset_canvas:
                st.session_state['canvas']['canvas_key'] = f'editor_{d}'
                st.session_state['canvas']['previous_objects'] = []

            st.session_state['canvas']['previous_d'] = d

            canvas_result = st_canvas(
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_image=get_overlay(image_slice, mask_slice, H, W, N, OVERLAY_COLORS),
                update_streamlit=True,
                height=H * DISPLAY_W / W,
                width=DISPLAY_W,
                drawing_mode='freedraw',
                key=st.session_state['canvas']['canvas_key']
            )

            # Track current objects
            current_objects = []
            if canvas_result and canvas_result.json_data:
                current_objects = canvas_result.json_data.get("objects", [])
            st.session_state['canvas']['previous_objects'] = current_objects

            # Save / clear buttons (trigger rerun only here)
            col_save, col_clear = st.columns([1, 0.3])
            with col_save:
                save_contour = st.button('Save Contour', type='primary', use_container_width=True)
                if save_contour and canvas_result and canvas_result.image_data is not None and current_objects:
                    brush_data = np.array(canvas_result.image_data)
                    rgb = brush_data[:, :, :3].astype(np.float32)
                    alpha = brush_data[:, :, 3].astype(np.float32) / 255.0

                    overlay_colors_list = np.array([color[:3] for color in OVERLAY_COLORS.values()], dtype=np.float32)
                    overlay_channels = list(OVERLAY_COLORS.keys())

                    h, w, _ = rgb.shape
                    rgb_flat = rgb.reshape(-1, 3)
                    alpha_flat = alpha.flatten()
                    distances = np.linalg.norm(rgb_flat[:, None, :] - overlay_colors_list[None, :, :], axis=-1)
                    closest_idx = np.argmin(distances, axis=1)

                    mask_flat = np.zeros((h * w, len(overlay_channels)), dtype=np.uint8)
                    for idx_color, ch in enumerate(overlay_channels):
                        mask_flat[:, idx_color] = ((closest_idx == idx_color) & (alpha_flat > 0)).astype(np.uint8)

                    masks = []
                    for idx_color, ch in enumerate(overlay_channels):
                        mask_bool = mask_flat[:, idx_color].reshape(h, w)
                        mask_bool = thicken_close_fill_and_smooth(mask_bool, stroke_width)
                        masks.append(mask_bool)

                    combined_mask = np.stack(masks, axis=-1)
                    for idx_color, ch in enumerate(overlay_channels):
                        resized_mask = np.array(
                            Image.fromarray(combined_mask[:, :, idx_color]).resize(
                                (W * st.session_state['subpixel_resolution'],
                                 H * st.session_state['subpixel_resolution']),
                                resample=Image.NEAREST
                            )
                        )
                        edited_mask[:, :, d, idx, :][resized_mask > 0] = 0
                        edited_mask[:, :, d, idx, ch][resized_mask > 0] = 1

                    st.session_state['edit_made'] = True
                    save_mask(edited_mask, save_path=st.session_state['cache_mask_path'])
                    st.rerun()

            with col_clear:
                if st.button('Clear Slice', use_container_width=True):
                    edited_mask[:, :, d, idx, :] = 0
                    save_mask(edited_mask, save_path=st.session_state['cache_mask_path'])

                    st.session_state['edit_made'] = True
                    st.rerun()

    # ---------- right column preview ----------
    with col3:
        view_mode = st.radio(
            "Corrected Mask",
            ["Static", "Viewer"],
            index=0,
            horizontal=True,
        )

        if st.session_state.get("edited_frames") is None or st.session_state["edit_made"]:
            make_video(
                image,
                edited_mask,
                save_file=full_edited_gif_path,
                mask_frames=[dia_idx, sys_idx],
            )
            gif = Image.open(full_edited_gif_path)
            st.session_state["edited_frames"] = [f.copy() for f in ImageSequence.Iterator(gif)]
            st.session_state["edit_made"] = False

        if view_mode == "Static":
            view_image = st.session_state["edited_frames"][0 if idx_label == "End-Diastole" else 1]
            width = int(DISPLAY_W * 1.5)
        elif view_mode == "Viewer":
            view_image = image_slice
            width = int(DISPLAY_W)

        st.image(view_image, width=width)
