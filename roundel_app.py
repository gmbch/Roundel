# --------------------------------------------------------------
# Configure Streamlit page
# --------------------------------------------------------------
from roundel_utils import *
from aws_utils import (
    fetch_staged_roundel_cases,
    download_sax_artifacts,
    save_masks_and_metrics,
    delete_intermediate_session,
    build_ambra_link,
)
from fourc_viewer import render_4c_review

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

    # ---------------------------------------------------
    # DETERMINE CURRENT SELECTED INDEX
    # ---------------------------------------------------

    selected_uid = st.session_state.get(
        "selected_study_uid",
        cases[0]["study_uid"]
    )

    case_uids = [c["study_uid"] for c in cases]

    selected_index = (
        case_uids.index(selected_uid)
        if selected_uid in case_uids
        else 0
    )

    selected_case = st.selectbox(
        "Select Case",
        options=cases,
        index=selected_index,
        format_func=lambda c: f"{c.get('fid', 'Unknown')} | {c.get('study_date', 'Unknown')}"
    )

    st.session_state["selected_study_uid"] = (
        selected_case["study_uid"]
    )

    st.session_state["selected_case"] = selected_case

# Extract metadata
case_item   = selected_case
study_uid   = case_item["study_uid"]
patient     = case_item.get("fid")
study_date  = case_item.get("study_date")
sagemaker_flags  = case_item.get("sm_flags")
gif_url = case_item.get('gif_url')
pipeline_edv = case_item.get('edv', '')
pipeline_esv = case_item.get('esv', '')
pipeline_mass = case_item.get('mass', '')
model_ind = case_item.get('model_ind', '2D')

if not gif_url.startswith("http"):
    gif_url = f"https://{gif_url}"


comments  = case_item.get("comments")
description = case_item.get("description", "Unknown")
try:
    ambra_link = build_ambra_link(study_uid)
except Exception as exc:
    ambra_link = None
    ambra_link_error = str(exc)
site_code = str(case_item.get("site", "") or "").strip().upper()
if not site_code:
    site_code = str(patient or "").strip()[:3].upper()

stored_4c_uid = case_item.get("selected_4c_series_uid")
session_4c_uid = st.session_state.get("fourc_selected_series_uid")
session_4c_study_uid = st.session_state.get("fourc_selection_study_uid")
fourc_series_uid = (
    session_4c_uid
    if session_4c_uid and session_4c_study_uid == study_uid
    else stored_4c_uid
)
fourc_description = (
    st.session_state.get("fourc_selected_description", "")
    if session_4c_uid == fourc_series_uid and session_4c_study_uid == study_uid
    else case_item.get("selected_4c_description", "")
)
fourc_selection_confirmed = bool(
    fourc_series_uid and (
        case_item.get("fourc_selection_confirmed", True)
        or (session_4c_uid == fourc_series_uid and session_4c_study_uid == study_uid)
    )
)
fourc_unavailable = (
    st.session_state.get("fourc_unavailable_study_uid") == study_uid
)

pixelspacing = float(case_item["pixelspacing"])
thickness    = float(case_item["thickness"])


# --------------------------------------------------------------
# Download artifacts from S3
# --------------------------------------------------------------
local_dir = download_sax_artifacts(study_uid, saxdf_bool=True, model_ind=model_ind)

sax_series_uid_list = get_sax_series_uid_list(local_dir)

# Initialize Roundel logic
initialize_app(local_dir, patient, study_date, study_uid, pixelspacing, thickness,  preprocess=True)
st.session_state["fourc_selected_series_uid"] = fourc_series_uid
st.session_state["fourc_selection_study_uid"] = study_uid
st.session_state["fourc_selection_confirmed"] = fourc_selection_confirmed
st.session_state["fourc_site_code"] = site_code
# --------------------------------------------------------------
# Display sidebar metadata
# --------------------------------------------------------------
with col2:
    # Display metadata in the app
    st.markdown(f"**🟢 {len(cases)} total staged cases**")
    st.markdown(f"**Study UID:** {study_uid} | **FID:** {patient} | **Study Date:** {study_date} | **Site:** {site_code} | **2D or 4D:** {model_ind}")
    st.markdown(f"**Flags from SageMaker:** {sagemaker_flags}")
    st.markdown(f"**SageMaker Comments:** {comments}")
    st.markdown(f"**Description:** {description} | **Pixel Size**: {pixelspacing} x {pixelspacing}mm | **Slice Thickness**: {thickness} mm")
    if ambra_link:
        st.markdown(f"[Open study in Ambra]({ambra_link})")
    else:
        st.caption(f"Ambra study link unavailable: {ambra_link_error}")
    if fourc_selection_confirmed:
        st.success(f"4C selected: `{fourc_series_uid}`")
    elif fourc_unavailable:
        st.info("No usable 4C series are available; continuing without 4C tools.")

    if case_item.get("checkpoint_exists"):
        st.warning("💾 Recoverable draft session exists")

    st.markdown(f"**Pipeline: EDV**: {pipeline_edv:.1f} mL | **ESV**: {pipeline_esv:.1f} mL | **Mass**: {pipeline_mass:.1f} g")
    st.markdown(f"[📥 Download Segmentation GIF]({gif_url})")


    # --- Skip Case Button ---
    if st.button("⏭️ Skip Case (Artifacts Too Heavy)", type="secondary", use_container_width=True):
        skip_case(study_uid, patient, study_date, skip_type='skip')
        cleanup_case_artifacts(
            study_uid=study_uid,
            local_dir=local_dir
        )

    # --- Push to Circle Case Button ---
    if st.button("⏭️ Needs Circle (Task Too Complex)", type="secondary", use_container_width=True):
        skip_case(study_uid, patient, study_date, skip_type='circle')
        cleanup_case_artifacts(
            study_uid=study_uid,
            local_dir=local_dir
        )

# --------------------------------------------------------------
# App
# --------------------------------------------------------------

available_views = ["4C Viewer 🫀", "EDV/ESV Finder 🔍", "Mask Editor 📝", "Final Result ✅"]
requested_view = st.session_state.pop("next_view", None)
if st.session_state.get("roundel_view_study_uid") != study_uid:
    st.session_state["roundel_view_study_uid"] = study_uid
    st.session_state["roundel_view"] = "4C Viewer 🫀"
if requested_view in available_views:
    st.session_state["roundel_view"] = requested_view

view = st.segmented_control(
    "Tab",
    options=available_views,
    key="roundel_view",
    label_visibility='hidden'
)
st.divider()

# --------------------------------------------------------------
# 4C Viewer
# --------------------------------------------------------------
if view == "4C Viewer 🫀":
    selected_4c_uid = render_4c_review(study_uid, site_code, case_item)
    if selected_4c_uid:
        st.session_state["fourc_selected_series_uid"] = selected_4c_uid
        st.session_state["fourc_selection_study_uid"] = study_uid
        st.session_state["fourc_selection_confirmed"] = True
        st.info("4C selection is complete. Open the Mask Editor when ready.")

# EDV/ESV Finder
# --------------------------------------------------------------
if view == "EDV/ESV Finder 🔍":
    edv_esv_view()

# --------------------------------------------------------------
# Mask Editor
# --------------------------------------------------------------


if view == "Mask Editor 📝":
    if not fourc_selection_confirmed and not fourc_unavailable:
        st.error("Complete and confirm the 4C Viewer selection first.")
        st.stop()
    mask_editor_view()

# --------------------------------------------------------------
# Final Result
# --------------------------------------------------------------

if view == "Final Result ✅":
    if not fourc_selection_confirmed and not fourc_unavailable:
        st.error("Complete and confirm the 4C Viewer selection first.")
        st.stop()
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

    edited_mask = cv_zoom(edited_mask,
                          zoom=[1 / st.session_state['subpixel_resolution'],
                                1 / st.session_state['subpixel_resolution'], 1, 1],
                          interpolation=cv2.INTER_NEAREST)

    final_gif_path = f'results/gifs/{study_uid}.gif'
    # Compute metrics
    volume, masses, edv, esv, sv, ef, mass = calculate_sax_metrics(
        edited_mask, pixelspacing, thickness, dia_idx, sys_idx)

    # Create full-size arrays
    final_mask_2d = np.zeros(raw_mask.shape, dtype=raw_mask.dtype)
    final_mask_2d[y_min:y_max, x_min:x_max, :, [dia_idx, sys_idx], 1:] = edited_mask[:, :, :, [dia_idx, sys_idx], 1:]
    final_mask_2d = np.argmax(final_mask_2d, -1)

    make_video(
        preprocessed_image,
        edited_mask,
        mask_frames=[dia_idx, sys_idx],
        save_file=edited_gif_path
    )

    make_video(
        raw_image,
        np.eye(N, dtype=np.uint8)[final_mask_2d],
        save_file=final_gif_path,
        mask_frames=[dia_idx, sys_idx],
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
                model_ind=model_ind,
                fourc_series_uid=fourc_series_uid,
                fourc_description=fourc_description
            )

            delete_intermediate_session(study_uid)

            cleanup_case_artifacts(
                study_uid=study_uid,
                local_dir=local_dir
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
                "edited_frames", "mask_hash", "cache_config_path", "cache_mask_path",
                "point1", "point2", "coord1", "coord2", "crop1", "crop2",
                "fourc_selected_series_uid", "fourc_selection_study_uid",
                "fourc_selection_confirmed", "fourc_candidate_series_uid",
                "fourc_candidate_study_uid", "fourc_selected_description",
                "fourc_unavailable_study_uid"
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

            st.session_state["selected_study_uid"] = (
                new_cases[0]["study_uid"]
            )

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
