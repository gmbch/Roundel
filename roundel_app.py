# --------------------------------------------------------------
# Configure Streamlit page
# --------------------------------------------------------------
from roundel_utils import *
from aws_utils import (
    fetch_staged_roundel_cases,
    download_sax_artifacts,
    save_masks_and_metrics
)

st.set_page_config(page_title="Roundel", page_icon="⭕️", layout='wide')

# --------------------------------------------------------------
# Fetch staged Roundel cases from DDB
# --------------------------------------------------------------
cases = fetch_staged_roundel_cases()
# Ensure FID/patient is sortable and then sort
cases = sorted(cases, key=lambda c: str(c.get("fid", "")))

# st.sidebar.success(f"")


if not cases:
    st.sidebar.info("No Roundel cases waiting.")
    st.stop()

# Sidebar selection (use case objects instead of uid list)
col1, col2 = st.columns([0.3, 0.7])
with col1:
    # selected_case = st.selectbox(
    #     "Select Case",
    #     options=cases,
    #     format_func=lambda c: f"{c.get('fid','Unknown')} | {c.get('study_date','Unknown')} | {c.get('study_uid')}"
    # )
    selected_case = st.selectbox(
        "Select Case",
        options=cases,
        index=cases.index(st.session_state.get("selected_case", cases[0])),
        format_func=lambda c: f"{c.get('fid', 'Unknown')} | {c.get('study_date', 'Unknown')}"
    )

    st.session_state["selected_case"] = selected_case

# Extract metadata
case_item   = selected_case
study_uid   = case_item["study_uid"]
patient     = case_item.get("fid")
study_date  = case_item.get("study_date")
description = case_item.get("description", "Unknown")

pixelspacing = float(case_item["pixelspacing"])
thickness    = float(case_item["thickness"])


# --------------------------------------------------------------
# Download artifacts from S3
# --------------------------------------------------------------
local_dir = download_sax_artifacts(study_uid, saxdf_bool=False)

# Initialize Roundel logic
initialize_app(local_dir, study_uid, pixelspacing, thickness,  preprocess=True)
# --------------------------------------------------------------
# Display sidebar metadata
# --------------------------------------------------------------
with col2:
    # Display metadata in the app
    st.markdown(f"**🟢 {len(cases)} total staged cases**")
    st.markdown(f"**Study UID:** {study_uid} | **FID:** {patient} | **Study Date:** {study_date}")
    st.markdown(f"**Description:** {description} | **Pixel Size**: {pixelspacing} x {pixelspacing}mm | **Slice Thickness**: {thickness} mm")

    # --- Skip Case Button ---
    if st.button("⏭️ Skip Case (Artifacts Too Heavy)", type="secondary", use_container_width=True):
        skip_case(study_uid, patient, study_date)

# --------------------------------------------------------------
# App
# --------------------------------------------------------------

# If the save flow set a next default tab, use it
if "next_view" in st.session_state:
    st.session_state["view"] = st.session_state.pop("next_view")


view = st.radio(
    "Tab",
    options=["EDV/ESV Finder 🔍", "Mask Editor 📝", "Final Result ✅"],
    horizontal=True,
    key="view"
)

st.divider()

# --------------------------------------------------------------
# EDV/ESV Finder
# --------------------------------------------------------------
if view == "EDV/ESV Finder 🔍":
    edv_esv_view()

# --------------------------------------------------------------
# Mask Editor
# --------------------------------------------------------------


if view == "Mask Editor 📝":
    try:
        mask_editor_view()
    except:
        st.rerun()

# --------------------------------------------------------------
# Final Result
# --------------------------------------------------------------

if view == "Final Result ✅":
    raw = st.session_state.raw
    preprocessed = st.session_state.preprocessed

    raw_image = raw["image"]
    raw_mask = raw["mask"]
    raw_edv = raw["raw_edv"]
    raw_esv = raw["raw_esv"]
    raw_mass = raw["raw_mass"]
    raw_ef = raw["raw_ef"]
    preprocessed_image = preprocessed["image"]
    H, W, D, T, N = [preprocessed[k] for k in ["H", "W", "D", "T", "N"]]

    edited_mask = st.session_state['edited_mask']
    x_min, y_min, x_max, y_max = preprocessed['crop_box']
    dia_idx = st.session_state.edv_esv_selected['dia_idx']
    sys_idx = st.session_state.edv_esv_selected['sys_idx']

    if not st.session_state.edv_esv_selected["confirmed"]:
        st.error("Select and confirm EDV/ESV first.")
        st.stop()

    edited_mask = cv_zoom(edited_mask, zoom=[1 / st.session_state['subpixel_resolution'],
                                             1 / st.session_state['subpixel_resolution'], 1, 1])

    final_gif_path = f'results/gifs/{study_uid}.gif'
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
        scale=1.5
    )

    col1, _, col2, col3 = st.columns([0.08, 0.05, 0.2, 0.3])
    with col1:
        st.caption('Metrics')
        st.metric("End-Diastolic Volume", f"{edv:.1f} mL",
                  delta=None if edv == raw_edv else f"{edv - raw_edv:.1f} mL")
        st.metric("End-Systolic Volume", f"{esv:.1f} mL",
                  delta=None if esv == raw_esv else f"{esv - raw_esv:.1f} mL")
        st.metric("Ejection Fraction", f"{ef:.1f} %",
                  delta=None if round(ef, 1) == round(raw_ef, 1) else f"{ef - raw_ef:.1f} %")
        st.metric("Myocardial Mass", f"{mass:.1f} g",
                  delta=None if mass == raw_mass else f"{mass - raw_mass:.1f} g")


        if st.button('Save Masks and Metrics', type='primary', use_container_width=True):
            print(study_uid)
            save_masks_and_metrics(
                study_uid=study_uid,
                pixelspacing=pixelspacing,
                thickness=thickness,
                dia_idx=dia_idx,
                sys_idx=sys_idx,
                edited_mask=st.session_state.edited_mask,
                final_mask_2d=final_mask_2d,
                edited_gif_path=edited_gif_path,
                full_gif_path=final_gif_path,
                raw_edv=raw_edv,
                raw_esv=raw_esv,
                raw_mass=raw_mass,
                raw_ef=raw_ef,
                metrics=(edv, esv, sv, ef, mass),
                patient=patient,
                study_date=study_date,
                description=description,
            )

            st.success(f"Saved results for {patient} ({study_date})")

            # ----------------------------------------------------------
            # Refresh staged cases
            # ----------------------------------------------------------
            new_cases = fetch_staged_roundel_cases()
            new_cases = sorted(new_cases, key=lambda c: str(c.get("fid", "")))

            # Remove current study from session if still present
            if "selected_case" in st.session_state:
                st.session_state.pop("selected_case", None)

            # ----------------------------------------------------------
            # Clear session state (fresh start)
            # ----------------------------------------------------------
            reset_keys = [
                "edited_mask", "edv_esv_selected", "preprocessed", "raw",
                "point1", "point2", "coord1", "coord2", "crop1", "crop2"
            ]
            for k in reset_keys:
                st.session_state.pop(k, None)

            # ----------------------------------------------------------
            # If no more cases → notify and stop
            # ----------------------------------------------------------
            if not new_cases:
                st.sidebar.success("🎉 All Roundel cases completed!")
                st.stop()

            # ----------------------------------------------------------
            # Select the next case automatically
            # ----------------------------------------------------------
            st.session_state["selected_case"] = new_cases[0]

            # Set tab for the NEXT run
            st.session_state["next_view"] = "EDV/ESV Finder 🔍"

            # Trigger app reload
            st.rerun()

    with col2:
        st.caption('Final Cropped Mask')
        st.image(edited_gif_path)

    with col3:
        st.caption('Final Full-Sized Mask')
        st.image(final_gif_path)
