import os, sys
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
    binary_erosion,
    binary_closing,
    gaussian_filter
)  
from skimage.morphology import disk  
from skimage.measure import find_contours
import pandas as pd
import time
import cv2

os.makedirs('results/temp', exist_ok=True)
os.makedirs('results/gifs', exist_ok=True)
os.makedirs('results/masks', exist_ok=True)
os.makedirs('results/edited_sax_df', exist_ok=True)

GIF_W = 150
DISPLAY_H = DISPLAY_W = 400
BACKGROUND_COLOR = (30, 30, 30, 0)
LV_MYO_COLOR = (0, 255, 255, 50) # Blue
LV_COLOR = (255, 10, 10, 50)      # Red

background_idx = 0
lv_myo_idx = 1
lv_idx = 2

channels = [lv_myo_idx, lv_idx]

BRUSH_LABELS = {
    lv_myo_idx: 'LV Myocardium 🔵',
    lv_idx: 'LV Blood Pool 🔴',
}


OVERLAY_COLORS = {
    background_idx: BACKGROUND_COLOR,
    lv_myo_idx: LV_MYO_COLOR,
    lv_idx: LV_COLOR,
}

blank_gif_path = f'results/temp/blank.gif'
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


    st.session_state['subpixel_resolution'] = 4
    # -----------------------------
    # Preprocess / crop if required
    # -----------------------------
    if preprocess:
        x_min, y_min, x_max, y_max = find_crop_box(np.max(raw_mask[...,[lv_idx, lv_myo_idx]], axis=(-1,-2,-3)), crop_factor=1.5)
        preprocessed_image = raw_image[y_min:y_max, x_min:x_max, :, :]
        preprocessed_mask = raw_mask[y_min:y_max, x_min:x_max, :, :, :].astype('uint8')
        H, W, D, T, N = preprocessed_mask.shape

        has_masks = np.where(np.sum(preprocessed_mask[...,-1], axis = (0,1,3))>0)[0]
        mid_slice = len(has_masks)//2
        make_video(preprocessed_image[:,:,has_masks[mid_slice-3:mid_slice+3],:], preprocessed_mask[:,:,has_masks[mid_slice-3:mid_slice+3],:, :] * 0, save_file=edv_esv_gif_path)


        smoothed_image = cv_zoom(preprocessed_image, zoom = [st.session_state['subpixel_resolution'],st.session_state['subpixel_resolution'],1,1])
        smoothed_mask = smooth_zoom(preprocessed_mask, zoom = [st.session_state['subpixel_resolution'],st.session_state['subpixel_resolution'],1,1,1])
        make_video(preprocessed_image, preprocessed_mask*0, save_file=blank_gif_path)
                
        # make_video(smoothed_image, smoothed_mask, save_file=preprocessed_gif_path)

        gif = Image.open(edv_esv_gif_path)

        st.session_state.preprocessed = {
            "image": preprocessed_image,
            "mask": preprocessed_mask,
            "smooth_image": smoothed_image,
            "smooth_mask": smoothed_mask,
            "H": H, "W": W, "D": D, "T": T, "N": N,
            "edv_esv_frames": [frame.copy() for frame in ImageSequence.Iterator(gif)],
            'crop_box':[x_min, y_min, x_max, y_max]
        }

    else:
        st.session_state['subpixel_resolution'] = 1
        st.session_state.preprocessed = {
            "image": raw_image,
            "mask": raw_mask,
            "H": raw_shape[0], "W": raw_shape[1], "D": raw_shape[2], "T": raw_shape[3], "N": N,
            "frames": None,
            'crop_box':[0, 0, raw_image.shape[0], raw_image.shape[1]]

        }

    # # -----------------------------
    # # Plot raw curve
    # # -----------------------------
    # plot_volume_curve(
    #     raw_volume, raw_volume,
    #     raw_dia_idx, raw_sys_idx, raw_dia_idx, raw_sys_idx,
    #     save_path=raw_curve_path
    # )


    # -----------------------------
    # Initialize edited mask
    # -----------------------------
    st.session_state['edited_mask'] = st.session_state.preprocessed["smooth_mask"].copy()
    st.session_state['mask_hash']= mask_hash(st.session_state.preprocessed["mask"])
    st.session_state.initialized_all = True


def cv_zoom(images, zoom=[4,4,1,1,1], interpolation=cv2.INTER_CUBIC):
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
        resized = np.zeros((int(h*h_zoom), int(w*w_zoom), d, t), dtype=images.dtype)
        for z in range(d):
            for tau in range(t):
                resized[..., z, tau] = cv2.resize(images[..., z, tau], (int(w*w_zoom), int(h*h_zoom)), interpolation=interpolation)
    elif images.ndim == 5:
        h, w, d, t, c = images.shape
        resized = np.zeros((int(h*h_zoom), int(w*w_zoom), d, t, c), dtype=images.dtype)
        for z in range(d):
            for tau in range(t):
                for ch in range(c):
                    resized[..., z, tau, ch] = cv2.resize(images[..., z, tau, ch], (int(w*w_zoom), int(h*h_zoom)), interpolation=interpolation)
    else:
        raise ValueError("Input must be 4D or 5D array.")

    return resized

def smooth_zoom(mask, zoom=[4,4,1,1,1], sigma=5.0, to_discrete=True):
    """
    Zoom a 4D or 5D categorical mask and smooth edges for visual appearance.

    Args:
        mask (np.ndarray): Input mask of shape H,W,D,T or H,W,D,T,C
        zoom (list): Zoom factors for H,W,D,T,(C). Only H and W >1
        sigma (float): Gaussian blur sigma
        to_discrete (bool): If True, round blurred mask back to original integer labels

    Returns:
        np.ndarray: Zoomed and smoothed mask
    """
    # Step 1: Zoom with nearest-neighbor to preserve labels
    zoomed = cv_zoom(mask.astype(np.float32), zoom, interpolation=cv2.INTER_CUBIC)
    dims = zoomed.ndim
    if dims == 4:
        H,W,D,T = zoomed.shape
        for z in range(D):
            for t in range(T):
                zoomed[..., z, t] = cv2.GaussianBlur(zoomed[..., z, t], (0,0), sigmaX=sigma, sigmaY=sigma)
    elif dims == 5:
        H,W,D,T,C = zoomed.shape
        for z in range(D):
            for t in range(T):
                for c in range(C):
                    zoomed[..., z, t, c] = cv2.GaussianBlur(zoomed[..., z, t, c], (0,0), sigmaX=sigma, sigmaY=sigma)
    else:
        raise ValueError("Mask must be 4D or 5D")

    # Step 2: Optionally convert back to integer labels
    if to_discrete:
        zoomed = np.rint(zoomed).astype(mask.dtype)

    return zoomed


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




def make_video(image, mask, save_file, mask_frames = 'all', scale=1):
    position = image.shape[2]
    timesteps = image.shape[3]

    grid_rows = int(np.sqrt(position) + 0.5)
    grid_cols = (position + grid_rows - 1) // grid_rows

    H, W = image.shape[:2]
    GIF_H = H*GIF_W/W
    H_scaled, W_scaled = round(GIF_H * scale), round(GIF_W * scale)
    img_min, img_max = np.min(image), np.max(image)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", int(18 * scale))
    except:
        font = ImageFont.load_default()

    frames = []
    if mask_frames == 'all':
        mask_frames = np.arange(timesteps)

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
            if t in mask_frames:
                for ch in channels:
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
    volume = np.sum(mask[..., lv_idx], axis=(0,1,2)) * voxel_size
    masses = np.sum(mask[..., lv_myo_idx], axis=(0,1,2)) * voxel_size * 1.05
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
        key = key,
        on_change=wrap,
        args=(key, 0, T-1),
        disabled=disabled_flag
    )
    st.image(frames[idx], use_container_width=True)
    return idx



def edv_esv_view():
    """Full EDV/ESV Finder view layout."""
    if "edv_esv_selected" not in st.session_state:
        st.session_state.edv_esv_selected = {"dia_idx": None, "sys_idx": None, "confirmed": False}

    frames= st.session_state.preprocessed['edv_esv_frames']
    raw_dia_idx=st.session_state.raw['raw_dia_idx']
    raw_sys_idx=st.session_state.raw['raw_sys_idx'] 
    H, W, D, T, N = [st.session_state.preprocessed[k] for k in ["H","W","D","T","N"]]

    disabled_flag = st.session_state.edv_esv_selected["confirmed"]

    _, col_center,_ = st.columns([0.2,0.6,0.2])
    with col_center:
        col_edv, _, col_esv = st.columns([0.45,0.1,0.45])

        with col_edv:
            dia_idx = frame_index_slider(T, frames, raw_dia_idx, 'EDV Index', disabled_flag, key = 'edv')

        with col_esv:
            sys_idx = frame_index_slider(T, frames, raw_sys_idx, 'ESV Index',disabled_flag, key = 'esv')

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
            st.session_state.preprocessed["smooth_image"][:, :, :, [dia_idx, sys_idx]],
            st.session_state.preprocessed["smooth_mask"][:, :, :, [dia_idx, sys_idx], :],
            save_file=edited_gif_path
        )


def slice_navigation(D):
    if "slice_idx" not in st.session_state:
        st.session_state.slice_idx = 0

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

    return st.session_state.slice_idx


def get_overlay(img_slice, mask_state, H, W, N, OVERLAY_COLORS):
    overlay = Image.fromarray(np.stack([img_slice]*3, axis=-1)).convert("RGBA")
    for i in channels:
        ch_mask = mask_state[:, :, i]
        if np.any(ch_mask):
            mask_img = np.zeros((H*st.session_state['subpixel_resolution'], W*st.session_state['subpixel_resolution'], 4), dtype=np.uint8)
            mask_img[ch_mask > 0] = OVERLAY_COLORS[i]
            overlay = Image.alpha_composite(overlay, Image.fromarray(mask_img))
    return overlay


def select_brush(N):
    """Brush selection UI for channel, action, and stroke width."""
    action = st.radio("Brush Stroke Selection", options=["Paint ✏️", "Erase ✂️"],  index=0, horizontal=True)
    # stroke_width_map = {"1":5,"2":10,"3":20,"4":30,"5":60}
    stroke_width_map = {"thin":6,"medium":20,"thick":40}

    stroke_width_sel = st.radio("Stroke width", options=list(stroke_width_map.keys()),  index= 0 if action == "Paint ✏️" else 2, horizontal=True)
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


def draw_segmentation(img_slice, mask_slice, H, W, N, OVERLAY_COLORS, stroke_width, stroke_color, action, d, idx, channel):

    if 'canvas' not in st.session_state:
        st.session_state['canvas'] = {
            'canvas_key': f'editor_{d}',
            'previous_d': d,
            'previous_objects': []
        }

    canvas_result = st_canvas(
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_image=get_overlay(img_slice, mask_slice, H, W, N, OVERLAY_COLORS),
        update_streamlit=True,
        height=DISPLAY_H,
        width=DISPLAY_W,
        drawing_mode='freedraw',
        key=st.session_state['canvas']['canvas_key']
    )

    current_objects = []
    if canvas_result is not None and canvas_result.json_data is not None:
        current_objects = canvas_result.json_data.get("objects", [])

    if (
        d != st.session_state['canvas']['previous_d']
        and st.session_state['canvas']['previous_objects']
    ):
        st.session_state['canvas']['canvas_key'] = f'editor_{d}'
        st.session_state['canvas']['previous_d'] = d
        st.session_state['canvas']['previous_objects'] = []
        st.rerun()

    st.session_state['canvas']['previous_objects'] = current_objects

    col1, col2= st.columns([1, 0.3])
    edited_mask = st.session_state['edited_mask']

    with col1:
        save_contour = st.button('Save Contour', type='primary', use_container_width=True)
        if canvas_result and canvas_result.image_data is not None:
            objects = canvas_result.json_data.get("objects", [])
            if save_contour and objects:
                # Original canvas image
                brush_data = np.array(canvas_result.image_data)  # Hc x Wc x 4 (RGBA)
                rgb = brush_data[:, :, :3].astype(np.float32)
                alpha = brush_data[:, :, 3].astype(np.float32) / 255.0

                overlay_colors_list = np.array([color[:3] for color in OVERLAY_COLORS.values()], dtype=np.float32)
                overlay_channels = list(OVERLAY_COLORS.keys())

                h, w, _ = rgb.shape
                rgb_flat = rgb.reshape(-1, 3)
                alpha_flat = alpha.flatten()

                # Map each pixel to closest overlay color
                distances = np.linalg.norm(rgb_flat[:, None, :] - overlay_colors_list[None, :, :], axis=-1)
                closest_idx = np.argmin(distances, axis=1)

                # Prepare masks at canvas resolution
                mask_flat = np.zeros((h * w, len(overlay_channels)), dtype=np.uint8)
                for idx_color, channel in enumerate(overlay_channels):
                    mask_flat[:, idx_color] = ((closest_idx == idx_color) & (alpha_flat > 0)).astype(np.uint8)

                # Reshape masks and apply stroke thickening
                masks = []
                for idx_color, channel in enumerate(overlay_channels):
                    mask_bool = mask_flat[:, idx_color].reshape(h, w)
                    mask_bool = thicken_close_fill_and_smooth(mask_bool, stroke_width)
                    masks.append(mask_bool)

                # Combine all masks into a single array at canvas resolution
                combined_mask = np.stack(masks, axis=-1)  # Hc x Wc x num_channels

                # Resize all masks once at the end to target size
                for idx_color, channel in enumerate(overlay_channels):
                    resized_mask = np.array(
                        Image.fromarray(combined_mask[:, :, idx_color]).resize((W*st.session_state['subpixel_resolution'], H*st.session_state['subpixel_resolution']), resample=Image.NEAREST)
                    )

                    # Clear affected pixels first
                    edited_mask[:, :, d, idx, :][resized_mask > 0] = 0
                    # Apply current channel
                    edited_mask[:, :, d, idx, channel][resized_mask > 0] = 1

                st.rerun()

    with col2:
        if st.button('Clear Slice', use_container_width=True):
            edited_mask[:,:,d,idx,:] = 0
            st.rerun()


def display_corrected_mask(image, image_slice, edited_mask, dia_idx, sys_idx, idx_label, edited_gif_path):
    """Display corrected mask as GIF or static frame, caching loaded frames."""
    view_mode = st.radio('Corrected Mask', ['GIF', 'Static', 'Viewer'], index=0, horizontal=True)

    # Get cached frames if available
    frames = st.session_state.get('gif_frames', None)
    current_hash = hashlib.md5(edited_mask.tobytes()).hexdigest()

    # Regenerate frames if mask changed or not cached
    if frames is None or st.session_state.get('mask_hash') != current_hash:
        make_video(
            image[:, :, :, [dia_idx, sys_idx]],
            edited_mask[:, :, :, [dia_idx, sys_idx], :],
            save_file=edited_gif_path
        )
        gif = Image.open(edited_gif_path)
        frames = [frame.copy() for frame in ImageSequence.Iterator(gif)]
        st.session_state.gif_frames = frames
        st.session_state['mask_hash']= current_hash

    view_idx = 0 if idx_label == "End-Diastole" else 1

    if view_mode == 'GIF':
        st.image(blank_gif_path)
    elif view_mode == 'Static':
        st.image(frames[view_idx])
    else:
        st.image(image_slice, width=DISPLAY_W)


def mask_editor_view():
    """Full Mask Editor layout."""
    if not st.session_state.edv_esv_selected["confirmed"]:
        st.error("Select and confirm EDV/ESV first.")
        st.stop()

    H, W, D, T, N = [st.session_state.preprocessed[k] for k in ["H","W","D","T","N"]]
    image=st.session_state.preprocessed["smooth_image"]
    edited_mask=st.session_state['edited_mask']
    dia_idx=st.session_state.edv_esv_selected["dia_idx"]
    sys_idx=st.session_state.edv_esv_selected["sys_idx"]

    col1, col2, col3 = st.columns([1,1.5,1.5])

    with col1:
        channel, action, stroke_width = select_brush(N)
        st.divider()
        idx_label = st.radio("Frame", options=["End-Diastole","End-Systole"],  index = 0, horizontal=True)
        d = slice_navigation(D)
        
    

    idx = dia_idx if idx_label=="End-Diastole" else sys_idx
    img_slice = ((image[:,:,d,idx] - image[:,:,d,idx].min()) / (image[:,:,d,idx].max() - image[:,:,d,idx].min()) * 255).astype(np.uint8)

    with col2:
        edit_mode = st.radio('Segmentation Editor',['Editor','Viewer'], index=0, horizontal=True)
        stroke_color = f"rgba{OVERLAY_COLORS[background_idx][:3]+(0.8,)}" if action == "Erase ✂️" else f"rgba{OVERLAY_COLORS[channel][:3]+(0.4,)}"
        if edit_mode == 'Viewer':
            st.image(img_slice, width=DISPLAY_W)
        else:
            draw_segmentation(img_slice, edited_mask[:,:,d,idx,:], H, W, N, OVERLAY_COLORS, stroke_width, stroke_color, action, d, idx, channel)

        
    with col3:
        display_corrected_mask(image, img_slice, edited_mask, dia_idx, sys_idx, idx_label, edited_gif_path)