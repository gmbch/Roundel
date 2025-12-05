# --------------------------------------------------------------
# Configure Streamlit page
# --------------------------------------------------------------
from roundel_app_biventricular_utils import *
st.set_page_config(page_title="Roundel (Biventricular)", page_icon="⭕️", layout='wide')


# key = 'x'
# T = 20

# st.slider(
#     "w",
#     -1,
#     T + 1,
#     value=0,
#     key=key,
#     on_change=wrap,
#     args=(key, 0, T),
# )

# -----------------------------
# Data paths and series info
# -----------------------------
data_path = './data/test'
sax_series_uid_list = sorted([uid.replace('image___','').split('/')[-1].replace('.nii.gz','') for uid in glob.glob(f'{data_path}/*') if 'image' in uid and 'bi' in uid])

# Sidebar dropdown
st.write('# Roundel App (2D Biventricular)')

col1, col2 = st.columns([0.3,0.7])
with col1:
    sax_series_uid = st.selectbox(
        "Select SAX Series UID",
        options=sax_series_uid_list,
        index=0  # optional: preselect the first UID
    )

initialize_app(data_path, sax_series_uid, N, preprocess=True)
pixelspacing, thickness = st.session_state.raw['pixelspacing'], st.session_state.raw['thickness']

with col2:

    ## GABE IS THERE A WAY TO GET THIS INFORMATION FROM THE SAX_SERIES UID, ITS JUST FOR VISUALS
    patient, study_date, description = 'AAA-IMATEST-1', '01-01-2020', 'short axis cine stack'

    # Display metadata in the app
    st.markdown(f"**SAX Series UID:** {sax_series_uid} | **Patient:** {patient} | **Study Date:** {study_date}")
    st.markdown(f"**Description:** {description} | **Pixel Size**: {pixelspacing} x {pixelspacing}mm | **Slice Thickness**: {thickness} mm")

# --------------------------------------------------------------
# App
# --------------------------------------------------------------

view = st.segmented_control(
    "Tab",
    options=["EDV/ESV Finder 🔍", "Mask Editor 📝", "Final Result ✅"],
    default = "EDV/ESV Finder 🔍",
    label_visibility='hidden'
)
mini_divider()

H, W, D, T, N = [st.session_state.preprocessed[k] for k in ["H","W","D","T","N"]]

# --------------------------------------------------------------
# EDV/ESV Finder 
# --------------------------------------------------------------
if view == "EDV/ESV Finder 🔍":
    edv_esv_view(edv_esv_frames= st.session_state.preprocessed['edv_esv_frames'],
                 raw_lv_dia_idx=st.session_state.raw['raw_lv_dia_idx'], 
                 raw_rv_dia_idx=st.session_state.raw['raw_rv_dia_idx'], 
                 raw_lv_sys_idx=st.session_state.raw['raw_lv_sys_idx'], 
                 raw_rv_sys_idx=st.session_state.raw['raw_rv_sys_idx'], 
                 T = T)


# --------------------------------------------------------------
# Mask Editor 
# --------------------------------------------------------------

if view == "Mask Editor 📝":
    mask_editor_view(
        N=N, D=D, H=H, W=W,
        image=st.session_state.preprocessed["image"],
        lv_dia_idx=st.session_state.edv_esv_selected["lv_dia_idx"],
        lv_sys_idx=st.session_state.edv_esv_selected["lv_sys_idx"],
        rv_dia_idx=st.session_state.edv_esv_selected["rv_dia_idx"],
        rv_sys_idx=st.session_state.edv_esv_selected["rv_sys_idx"],
        OVERLAY_COLORS=OVERLAY_COLORS
    )


# --------------------------------------------------------------
# Final Result
# --------------------------------------------------------------
if view == "Final Result ✅":
    if not st.session_state.edv_esv_selected["confirmed"]:
        st.error("Select and confirm EDV/ESV first.")
        st.stop()

    raw_image = st.session_state.raw["image"]
    raw_mask = st.session_state.raw["mask"]
    preprocessed_image = st.session_state.preprocessed["image"]

    crop_box = st.session_state.preprocessed["crop_box"]

    raw_lv_dia_idx = st.session_state.raw["raw_lv_dia_idx"]
    raw_lv_sys_idx = st.session_state.raw["raw_lv_sys_idx"]
    raw_rv_dia_idx = st.session_state.raw["raw_rv_dia_idx"]
    raw_rv_sys_idx = st.session_state.raw["raw_rv_sys_idx"]

    lv_dia_idx = st.session_state['edv_esv_selected']["lv_dia_idx"]
    lv_sys_idx = st.session_state['edv_esv_selected']["lv_sys_idx"]
    rv_dia_idx = st.session_state['edv_esv_selected']["rv_dia_idx"]
    rv_sys_idx = st.session_state['edv_esv_selected']["rv_sys_idx"]

    final_lv_gif_path = f"results/gifs/{sax_series_uid}_lv.gif"
    final_rv_gif_path = f"results/gifs/{sax_series_uid}_rv.gif"

    # Calculate LV metrics
    lv_mask = st.session_state.edited_mask_lv
    lv_volume, lv_masses, lv_edv, lv_esv, lv_sv, lv_ef, lv_mass = calculate_sax_metrics(
        mask=lv_mask,
        pixelspacing=pixelspacing,
        thickness=thickness,
        blood_pool_idx=lv_idx,
        myo_idx=lv_myo_idx,
        dia_idx=lv_dia_idx,
        sys_idx=lv_sys_idx
    )
    raw_lv_volume, raw_lv_masses, raw_lv_edv, raw_lv_esv, raw_lv_sv, raw_lv_ef, raw_lv_mass = calculate_sax_metrics(
        mask=raw_mask,
        pixelspacing=pixelspacing,
        thickness=thickness,
        blood_pool_idx=lv_idx,
        myo_idx=lv_myo_idx,
        dia_idx=raw_lv_dia_idx,
        sys_idx=raw_lv_sys_idx
    )

    # Calculate RV metrics
    rv_mask = st.session_state.edited_mask_rv
    rv_volume, rv_masses, rv_edv, rv_esv, rv_sv, rv_ef, rv_mass = calculate_sax_metrics(
        mask=rv_mask,
        pixelspacing=pixelspacing,
        thickness=thickness,
        blood_pool_idx=rv_idx,
        myo_idx=rv_myo_idx,
        dia_idx=rv_dia_idx,
        sys_idx=rv_sys_idx
    )
    raw_rv_volume, raw_rv_masses, raw_rv_edv, raw_rv_esv, raw_rv_sv, raw_rv_ef, raw_rv_mass = calculate_sax_metrics(
        mask=raw_mask,
        pixelspacing=pixelspacing,
        thickness=thickness,
        blood_pool_idx=rv_idx,
        myo_idx=rv_myo_idx,
        dia_idx=raw_rv_dia_idx,
        sys_idx=raw_rv_sys_idx
    )

    # For LV
    final_lv_mask = resize_to_original(
        edited_mask=lv_mask,
        raw_mask=raw_mask,
        crop_box=crop_box,
        dia_idx=lv_dia_idx,
        sys_idx=lv_sys_idx
    )

    # For RV
    final_rv_mask = resize_to_original(
        edited_mask=rv_mask,
        raw_mask=raw_mask,
        crop_box=crop_box,
        dia_idx=rv_dia_idx,
        sys_idx=rv_sys_idx
    )

    x_min, y_min, x_max, y_max = crop_box


    # Generate final GIFs
    make_video(preprocessed_image[:,:,:,[lv_dia_idx,lv_sys_idx]], final_lv_mask[y_min:y_max,x_min:x_max,:,[lv_dia_idx,lv_sys_idx]], save_file=final_lv_gif_path, ventricle = 'lv')
    make_video(preprocessed_image[:,:,:,[rv_dia_idx,rv_sys_idx]], final_rv_mask[y_min:y_max,x_min:x_max,:,[rv_dia_idx,rv_sys_idx]], save_file=final_rv_gif_path, ventricle = 'rv')

    col_lv, col_rv = st.columns(2)
    with col_lv:
        st.markdown('#### Left Ventricle')

        col1, col2 = st.columns([0.2,0.8])
        with col1:
            st.caption("LV Metrics")
            st.metric("EDV", f"{lv_edv:.1f} mL", delta=format_delta(lv_edv, raw_lv_edv, " mL"))
            st.metric("ESV", f"{lv_esv:.1f} mL", delta=format_delta(lv_esv, raw_lv_esv, " mL"))
            st.metric("EF", f"{lv_ef:.1f} %", delta=format_delta(lv_ef, raw_lv_ef, " %", round_digits=1))
            st.metric("Mass", f"{lv_mass:.1f} g", delta=format_delta(lv_mass, raw_lv_mass, " g"))

        with col2:
            st.caption("Final LV Mask")
            st.image(final_lv_gif_path)
        
    with col_rv:
        st.markdown('#### Right Ventricle')

        col1, col2 = st.columns([0.2,0.8])
        with col1:
            st.caption("RV Metrics")
            st.metric("EDV", f"{rv_edv:.1f} mL", delta=format_delta(rv_edv, raw_rv_edv, " mL"))
            st.metric("ESV", f"{rv_esv:.1f} mL", delta=format_delta(rv_esv, raw_rv_esv, " mL"))
            st.metric("EF", f"{rv_ef:.1f} %", delta=format_delta(rv_ef, raw_rv_ef, " %", round_digits=1))
            st.metric("Mass", f"{rv_mass:.1f} g", delta=format_delta(rv_mass, raw_rv_mass, " g"))


        with col2:
            st.caption("Final RV Mask")
            st.image(final_rv_gif_path)

    
    if st.button('Save Masks and Metrics', type='primary', use_container_width=True):
        st.success('Masks and Metrics Saved!')

        # Save LV mask
        nib_lv_mask = nib.Nifti1Image(final_lv_mask, affine=np.eye(4), dtype='uint8')
        nib.save(nib_lv_mask, f'results/masks/{sax_series_uid}_lv.nii.gz')

        # Save RV mask
        nib_rv_mask = nib.Nifti1Image(final_rv_mask, affine=np.eye(4), dtype='uint8')
        nib.save(nib_rv_mask, f'results/masks/{sax_series_uid}_rv.nii.gz')

        # Prepare LV metrics
        lv_df = pd.DataFrame({
            "sax_series_uid": [sax_series_uid],
            "chamber": ["LV"],
            "edv_frame": [lv_dia_idx],
            "esv_frame": [lv_sys_idx],
            "edv": [lv_edv],
            "esv": [lv_esv],
            "stroke_volume": [lv_sv],
            "ejection_fraction": [lv_ef],
            "mass": [lv_mass],
            "pixelspacing": [pixelspacing],
            "thickness": [thickness],
            "num_slices": [lv_mask.shape[2]],
            "num_frames": [lv_mask.shape[3]],
        })

        # Prepare RV metrics
        rv_df = pd.DataFrame({
            "sax_series_uid": [sax_series_uid],
            "chamber": ["RV"],
            "edv_frame": [rv_dia_idx],
            "esv_frame": [rv_sys_idx],
            "edv": [rv_edv],
            "esv": [rv_esv],
            "stroke_volume": [rv_sv],
            "ejection_fraction": [rv_ef],
            "mass": [rv_mass],
            "pixelspacing": [pixelspacing],
            "thickness": [thickness],
            "num_slices": [rv_mask.shape[2]],
            "num_frames": [rv_mask.shape[3]],
        })

        # Combine LV and RV metrics
        combined_df = pd.concat([lv_df, rv_df], ignore_index=True)
        combined_df.to_csv(f'results/edited_sax_df/{sax_series_uid}.csv', index=False)

