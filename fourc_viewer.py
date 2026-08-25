"""Reusable 4C candidate review for the AWS-backed Roundel workflow."""

import io
import os
from decimal import Decimal
from pathlib import Path
import importlib.util

import boto3
import numpy as np
import pydicom
import streamlit as st
from boto3.dynamodb.conditions import Key

from aws_utils import table as roundel_table


AMBRA_BUCKET = os.getenv("AMBRA_SOURCE_BUCKET", "ambra-data-lake")
MIND_MAP_TABLE = os.getenv("MIND_MAP_TABLE", "dcm-mind-map")

s3 = boto3.client("s3")
dynamodb = boto3.resource("dynamodb")
mind_map_table = dynamodb.Table(MIND_MAP_TABLE)

_ALIGN_MODULE = None


def _to_decimal(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return Decimal(str(round(value, 6)))
    if isinstance(value, int):
        return Decimal(str(value))
    if isinstance(value, list):
        return [_to_decimal(v) for v in value]
    if isinstance(value, dict):
        return {key: _to_decimal(item) for key, item in value.items()}
    return value


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def is_candidate_4c_series(item):
    flow = str(item.get("Flow", "")).strip().upper() == "Y"
    cine = str(item.get("Cine", "")).strip().upper() == "Y"
    slices = _as_float(item.get("Slices"))
    series_type = str(item.get("Type", "")).strip().lower()
    return not flow and (series_type == "cine single" or (cine and slices == 1.0))


@st.cache_data(show_spinner=False)
def fetch_study_series(study_uid):
    rows = []
    response = mind_map_table.query(KeyConditionExpression=Key("study-uid").eq(study_uid))
    rows.extend(response.get("Items", []))
    while response.get("LastEvaluatedKey"):
        response = mind_map_table.query(
            KeyConditionExpression=Key("study-uid").eq(study_uid),
            ExclusiveStartKey=response["LastEvaluatedKey"],
        )
        rows.extend(response.get("Items", []))
    return sorted(rows, key=lambda row: str(row.get("Description", "")).lower())


def _first_value(value):
    if isinstance(value, (list, tuple)):
        return value[0]
    return value


def _window_image(pixel_array, dataset):
    arr = pixel_array.astype(np.float32)
    slope = _as_float(getattr(dataset, "RescaleSlope", 1.0)) or 1.0
    intercept = _as_float(getattr(dataset, "RescaleIntercept", 0.0)) or 0.0
    arr = arr * slope + intercept

    center = _as_float(_first_value(getattr(dataset, "WindowCenter", None)))
    width = _as_float(_first_value(getattr(dataset, "WindowWidth", None)))
    if center is not None and width not in (None, 0):
        low, high = center - width / 2.0, center + width / 2.0
    else:
        low, high = np.nanpercentile(arr, 1), np.nanpercentile(arr, 99)
        if not np.isfinite(low) or not np.isfinite(high) or low == high:
            low, high = np.nanmin(arr), np.nanmax(arr)
        if not np.isfinite(low) or not np.isfinite(high) or low == high:
            high = low + 1.0

    arr = np.clip(arr, low, high)
    arr = (arr - low) / max(high - low, 1e-6)
    if str(getattr(dataset, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        arr = 1.0 - arr
    return (arr * 255).astype(np.uint8)


def _extract_preview_frame(dataset):
    pixels = dataset.pixel_array
    while pixels.ndim > 2:
        pixels = pixels[pixels.shape[0] // 2]
    return _window_image(pixels, dataset)


@st.cache_data(show_spinner=False)
def fetch_series_preview(site_code, study_uid, series_uid):
    prefix = f"{site_code}/{study_uid}/{series_uid}/"
    keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=AMBRA_BUCKET, Prefix=prefix):
        keys.extend(
            obj["Key"] for obj in page.get("Contents", [])
            if obj["Key"].lower().endswith(".dcm")
        )
    if not keys:
        raise FileNotFoundError(f"No DICOM files found under s3://{AMBRA_BUCKET}/{prefix}")

    first_key = sorted(keys)[0]
    payload = s3.get_object(Bucket=AMBRA_BUCKET, Key=first_key)["Body"].read()
    dataset = pydicom.dcmread(io.BytesIO(payload), force=True)
    return {
        "image": _extract_preview_frame(dataset),
        "s3_key": first_key,
        "instance_number": getattr(dataset, "InstanceNumber", None),
    }


def _load_alignment_module():
    global _ALIGN_MODULE
    if _ALIGN_MODULE is not None:
        return _ALIGN_MODULE
    path = Path(__file__).parent / "roundel-vMW-2" / "align_4ch.py"
    spec = importlib.util.spec_from_file_location("roundel_align_4ch", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load 4CH alignment renderer from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _ALIGN_MODULE = module
    return module


def _prepare_alignment_sources(site_code, study_uid, series_uid, local_dir):
    """Materialize the selected cine and rich SAX geometry for align_4ch.py."""
    root = Path(local_dir) / "fourc" / study_uid / series_uid
    root.mkdir(parents=True, exist_ok=True)
    dcm_paths = sorted(root.glob("*.dcm"))
    if not dcm_paths:
        prefix = f"{site_code}/{study_uid}/{series_uid}/"
        keys = []
        for page in s3.get_paginator("list_objects_v2").paginate(
            Bucket=AMBRA_BUCKET, Prefix=prefix
        ):
            keys.extend(
                obj["Key"] for obj in page.get("Contents", [])
                if not obj["Key"].endswith("/")
            )
        for index, key in enumerate(sorted(keys)):
            target = root / f"{index:06d}.dcm"
            # Write the object directly. boto3's managed transfer uses a
            # temporary ``.dcm.<random>`` sibling, which can race with Dropbox
            # filesystem synchronization in this workspace.
            payload = s3.get_object(Bucket=AMBRA_BUCKET, Key=key)["Body"].read()
            target.write_bytes(payload)
        dcm_paths = sorted(root.glob("*.dcm"))

    from aws_utils import load_saxdf

    sax_df = load_saxdf(local_dir, study_uid)
    required = {"orientation", "position"}
    if not required.issubset(sax_df.columns):
        raise ValueError(
            "The downloaded SAX metadata does not contain orientation and position geometry."
        )
    sax_csv = Path(local_dir) / f"saxdf___{study_uid}_geometry.csv"
    sax_df.to_csv(sax_csv, index=False)
    return str(root), str(sax_csv)


def render_4ch_alignment(slice_idx, sax_frame, sax_nframes, local_dir):
    """Render the selected 4C cine with the current SAX slice intersection."""
    study_uid = st.session_state.get("fourc_selection_study_uid")
    series_uid = st.session_state.get("fourc_selected_series_uid")
    site_code = st.session_state.get("fourc_site_code")
    if not study_uid or not series_uid or not site_code:
        st.info("Confirm a 4C series before opening the cross-reference view.")
        return False

    try:
        fourch_dir, sax_csv = _prepare_alignment_sources(
            site_code, study_uid, series_uid, local_dir
        )
        st.session_state["force_4ch_dir"] = fourch_dir
        st.session_state["force_sax_csv"] = sax_csv
        align = _load_alignment_module()
        align.render_4ch_view(
            slice_idx, sax_frame=sax_frame, sax_nframes=sax_nframes,
            case={"study_uid": study_uid, "series_uid": series_uid},
        )
        return True
    except Exception as exc:
        st.warning(f"4CH cross-reference is unavailable for this case: {exc}")
        return False


def save_selected_series(study_uid, site_code, series_row, preview_info, candidate_count):
    roundel_table.update_item(
        Key={"study_uid": study_uid},
        UpdateExpression=(
            "SET selected_4c_series_uid = :series_uid, "
            "selected_4c_description = :description, "
            "selected_4c_group = :group_name, "
            "selected_4c_s3_key = :s3_key, "
            "selected_4c_site = :site_code, "
            "candidate_4c_series_count = :candidate_count, "
            "fourc_selection_confirmed = :confirmed"
        ),
        ExpressionAttributeValues=_to_decimal({
            ":series_uid": series_row["series-uid"],
            ":description": series_row.get("Description", ""),
            ":group_name": series_row.get("Group", ""),
            ":s3_key": preview_info["s3_key"],
            ":site_code": site_code,
            ":candidate_count": candidate_count,
            ":confirmed": True,
        }),
    )


def render_4c_review(study_uid, site_code, case_item):
    """Render the 4C review and return the confirmed series UID, if any."""
    st.session_state.pop("fourc_unavailable_study_uid", None)
    if not site_code:
        st.error("This case has no site metadata, so the Ambra DICOM path cannot be resolved.")
        st.session_state["fourc_unavailable_study_uid"] = study_uid
        return None

    all_series = fetch_study_series(study_uid)
    candidates = [row for row in all_series if is_candidate_4c_series(row)]
    if not candidates:
        st.info(
            "No eligible Cine Single series were found for this study. "
            "The 4C tools are unavailable, but the rest of the Roundel workflow remains available."
        )
        st.session_state["fourc_unavailable_study_uid"] = study_uid
        return None

    st.caption(f"Found {len(candidates)} eligible Cine Single series at site `{site_code}`.")
    preview_cache = {}
    preview_images = []
    preview_rows = []
    for row in candidates:
        uid = row["series-uid"]
        try:
            preview = fetch_series_preview(site_code, study_uid, uid)
        except Exception as exc:
            preview_cache[uid] = {"row": row, "preview": None, "error": str(exc)}
            continue
        preview_cache[uid] = {"row": row, "preview": preview, "error": None}
        preview_images.append(preview["image"])
        preview_rows.append(row)

    if not preview_images:
        st.info(
            "Eligible Cine Single series were found, but no DICOM previews could be loaded. "
            "The 4C tools are unavailable, but the rest of the Roundel workflow remains available."
        )
        st.session_state["fourc_unavailable_study_uid"] = study_uid
        return None

    selected_uid = st.session_state.get("fourc_candidate_series_uid")
    if not selected_uid and st.session_state.get("fourc_selection_study_uid") == study_uid:
        selected_uid = st.session_state.get("fourc_selected_series_uid")
    candidate_study_uid = st.session_state.get("fourc_candidate_study_uid")
    if candidate_study_uid and candidate_study_uid != study_uid:
        selected_uid = None
    try:
        from streamlit_image_select import image_select

        selected_index = image_select(
            "Select the 4C series",
            images=preview_images,
            captions=[str(row.get("Description") or row["series-uid"]) for row in preview_rows],
            index=0,
            key=f"fourc_grid_{study_uid}",
        )
        if selected_index is not None:
            selected_uid = preview_rows[int(selected_index)]["series-uid"]
            st.session_state["fourc_candidate_series_uid"] = selected_uid
            st.session_state["fourc_candidate_study_uid"] = study_uid
    except (ImportError, TypeError, ValueError):
        st.info("Install `streamlit-image-select` for direct image clicks; use the buttons below otherwise.")
        columns = st.columns(5)
        for index, row in enumerate(preview_rows):
            uid = row["series-uid"]
            with columns[index % 3]:
                st.image(preview_cache[uid]["preview"]["image"], use_container_width=True, clamp=True)
                st.caption(str(row.get("Description") or uid))
                if st.button("Choose", key=f"fourc_choose_{uid}", use_container_width=True):
                    selected_uid = uid
                    st.session_state["fourc_candidate_series_uid"] = uid
                    st.session_state["fourc_candidate_study_uid"] = study_uid

    for uid, payload in preview_cache.items():
        if payload["error"]:
            st.warning(f"Preview unavailable for `{uid}`: {payload['error']}")

    if not selected_uid or selected_uid not in preview_cache:
        st.info("Select a preview to continue. The Mask Editor is locked until this step is confirmed.")
        return None

    selected = preview_cache[selected_uid]
    st.subheader("Selected 4C Series")
    st.write(
        f"**Description:** {selected['row'].get('Description', 'Unknown')}  "
        f"\n**Series UID:** {selected_uid}  "
        f"\n**Group:** {selected['row'].get('Group', 'Unknown')}"
    )
    confirm = st.radio(
        "Are you sure you want to select this series for the 4C viewer?",
        ["No", "Yes"], index=0, horizontal=True, key=f"fourc_confirm_{study_uid}",
    )
    if st.button("Confirm 4C Selection", type="primary", disabled=confirm != "Yes"):
        save_selected_series(
            study_uid, site_code, selected["row"], selected["preview"], len(candidates)
        )
        st.session_state["fourc_selected_series_uid"] = selected_uid
        st.session_state["fourc_selected_description"] = selected["row"].get("Description", "")
        st.session_state["fourc_candidate_series_uid"] = selected_uid
        st.session_state["fourc_candidate_study_uid"] = study_uid
        st.session_state["fourc_selection_confirmed"] = True
        st.session_state["next_view"] = "EDV/ESV Finder 🔍"
        st.success(f"Confirmed `{selected['row'].get('Description', selected_uid)}` for 4C use.")
        st.rerun()

    return selected_uid if st.session_state.get("fourc_selection_confirmed") else None
