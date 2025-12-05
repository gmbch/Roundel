# --------------------------------------------------------------
# Configure Streamlit page
# --------------------------------------------------------------
from roundel_utils import *
st.set_page_config(page_title="Roundel", page_icon="⭕️", layout='wide')

# -----------------------------
# Data paths and series info
# -----------------------------
data_path = '/workspaces/Roundel/data/major_revisions'
# data_path = '/workspaces/local/Roundel/data/test'
sax_series_uid_list = sorted([uid.replace('image___','').split('/')[-1].replace('.nii.gz','') for uid in glob.glob(f'{data_path}/*') if 'image' in uid and 'bi' not in uid])
# Sidebar dropdown
st.write('# Roundel App (2D)')

col1, col2 = st.columns([0.3,0.7])
with col1:
    sax_series_uid = st.selectbox(
        "Select SAX Series UID",
        options=sax_series_uid_list,
        index=0  # optional: preselect the first UID
    )

initialize_app(data_path, sax_series_uid, preprocess=True)
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
st.divider()

H, W, D, T, N = [st.session_state.preprocessed[k] for k in ["H","W","D","T","N"]]

# --------------------------------------------------------------
# EDV/ESV Finder 
# --------------------------------------------------------------
if view == "EDV/ESV Finder 🔍":
    edv_esv_view(raw_curve_path = raw_curve_path, 
                 frames= st.session_state.preprocessed['edv_esv_frames'],
                 raw_dia_idx=st.session_state.raw['raw_dia_idx'], 
                 raw_sys_idx=st.session_state.raw['raw_sys_idx'], 
                 T = T, 
                 edited_gif_path=edited_gif_path)


# --------------------------------------------------------------
# Mask Editor 
# --------------------------------------------------------------

if view == "Mask Editor 📝":
    mask_editor_view(
        N=N, D=D, H=H, W=W,
        image=st.session_state.preprocessed["image"],
        edited_mask=st.session_state.edited_mask,
        dia_idx=st.session_state.edv_esv_selected["dia_idx"],
        sys_idx=st.session_state.edv_esv_selected["sys_idx"],
        OVERLAY_COLORS=OVERLAY_COLORS,
        edited_gif_path=edited_gif_path
    )

# --------------------------------------------------------------
# Final Result
# --------------------------------------------------------------

if view == "Final Result ✅":
    raw_image=st.session_state.raw['image']
    raw_mask=st.session_state.raw['mask']
    raw_edv = st.session_state.raw['raw_edv']
    raw_esv = st.session_state.raw['raw_esv']
    raw_mass = st.session_state.raw['raw_mass']
    raw_ef = st.session_state.raw['raw_ef']
    raw_shape=st.session_state.raw['shape']
    preprocessed_image=st.session_state.preprocessed['image']

    edited_mask=st.session_state.edited_mask
    x_min, y_min, x_max, y_max =st.session_state.preprocessed['crop_box']
    dia_idx=st.session_state.edv_esv_selected['dia_idx']
    sys_idx=st.session_state.edv_esv_selected['sys_idx']
    if not st.session_state.edv_esv_selected["confirmed"]:
        st.error("Select and confirm EDV/ESV first.")
        st.stop()

    final_gif_path = f'results/gifs/{sax_series_uid}.gif'
    # Compute metrics
    volume, masses, edv, esv, sv, ef, mass = calculate_sax_metrics(
        edited_mask, pixelspacing, thickness, dia_idx, sys_idx)

    # Create full-size arrays
    final_mask_2d = np.zeros(raw_mask.shape, dtype=raw_mask.dtype)
    final_mask_2d[y_min:y_max, x_min:x_max, :, [dia_idx, sys_idx], 1:] = edited_mask[:, :, :, [dia_idx, sys_idx], 1:]
    final_mask_2d = np.argmax(final_mask_2d, -1)
    
    make_video(
        preprocessed_image[:, :, :, [dia_idx, sys_idx]],
        edited_mask[:, :, :, [dia_idx, sys_idx], :],
        save_file=edited_gif_path
    )

    make_video(
        raw_image[:, :, :, [dia_idx, sys_idx]],
        np.eye(N, dtype=np.uint8)[final_mask_2d][:, :, :, [dia_idx, sys_idx], :],
        save_file=final_gif_path, 
        scale = 1.5
    )

    col1, _, col2, col3 = st.columns([0.075, 0.05,0.2, 0.3])
    with col1:
        st.caption('Metrics')
        st.metric("End-Diastolic Volume", f"{edv:.1f} mL",
                  delta=None if edv==raw_edv else f"{edv-raw_edv:.1f} mL")
        st.metric("End-Systolic Volume", f"{esv:.1f} mL",
                  delta=None if esv==raw_esv else f"{esv-raw_esv:.1f} mL")
        st.metric("Ejection Fraction", f"{ef:.1f} %",
                  delta=None if round(ef,1)==round(raw_ef,1) else f"{ef-raw_ef:.1f} %")
        st.metric("Myocardial Mass", f"{mass:.1f} g",
                  delta=None if mass==raw_mass else f"{mass-raw_mass:.1f} g")


        if st.button('Save Masks and Metrics', type='primary', use_container_width=True):
            st.success('Masks and Metrics Saved!')
            nib_mask = nib.Nifti1Image(final_mask_2d, affine=np.eye(4), dtype='uint8')
            nib.save(nib_mask, f'results/masks/{sax_series_uid}.nii.gz')

            df = pd.DataFrame({
                "sax_series_uid": [sax_series_uid],
                "edv_frame": [dia_idx],
                "esv_frame": [sys_idx],

                "edv": [edv],
                "esv": [esv],
                "stroke_volume": [sv],
                "ejection_fraction": [ef],
                "mass": [mass],

                "pixelspacing": [pixelspacing],
                "thickness": [thickness],

                "num_slices": [edited_mask.shape[2]],
                "num_frames": [edited_mask.shape[3]],
                }).to_csv(f'results/edited_sax_df/{sax_series_uid}.csv', index = False)

    with col2:
        st.caption('Final Cropped Mask')
        st.image(edited_gif_path)
    

    with col3:
        st.caption('Final Full-Sized Mask')
        st.image(final_gif_path)