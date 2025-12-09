import os
import boto3
import pandas as pd
import pickle
import nibabel as nib
import numpy as np
from datetime import datetime, timezone
from decimal import Decimal
import pydicom

# ------------------------------------------------------
# AWS Setup
# ------------------------------------------------------
ARTIFACT_BUCKET = os.getenv("ROUNDL_ARTIFACT_BUCKET", "dcmlab-img-storage")
ARTIFACT_PREFIX = os.getenv("ROUNDL_ARTIFACT_PREFIX", "ventricular sax_ssfp compress/")
SAXDF_PREFIX = os.getenv("ROUNDL_ARTIFACT_PREFIX", "ventricular sax_ssfp/")

RESULTS_BUCKET = os.getenv("ROUNDL_RESULTS_BUCKET", "dcmpipe-streamlit")
ROUNDEL_RESULTS_PREFIX = "roundel-results/"

STUDY_META_TABLE = os.getenv("ROUNDL_META_TABLE", "ambra-data-lake-study-meta-table")
ROUNDL_TABLE = os.getenv("ROUNDL_SESSIONS_TABLE", "roundel-sessions")

s3 = boto3.client("s3")
dynamodb = boto3.resource("dynamodb")
study_meta_table = dynamodb.Table(STUDY_META_TABLE)
sop_tbl = dynamodb.Table("ambra-data-lake-sop-meta-table")
table = dynamodb.Table(ROUNDL_TABLE)


def fetch_staged_roundel_cases():
    """
    DynamoDB table should contain entries like:
      {
          status: "staged",
          study_uid: "...",
          site: "...",
          fid: "...",
          (maybe other tracking metadata)
      }
    """
    table = dynamodb.Table(os.getenv("ROUNDL_SESSIONS_TABLE", "roundel-sessions"))

    resp = table.scan(
        FilterExpression="#s = :s",
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues={":s": "staged"}
    )

    return resp.get("Items", [])


def download_sax_artifacts(study_uid, saxdf_bool=True):
    """
    Download Roundel artifacts for a single SAX series.

    Expected S3 structure (flat layout):
        ventricular sax_ssfp/saxdf___{uid}.pkl
        ventricular sax_ssfp compress/image___{uid}.nii.gz
        ventricular sax_ssfp compress/masks___{uid}.nii.gz
    """
    local_dir = "./data/"
    os.makedirs(local_dir, exist_ok=True)

    # ----------------------------
    # Construct S3 keys
    # ----------------------------
    saxdf_key  = f"ventricular sax_ssfp/saxdf___{study_uid}.pkl"
    image_key  = f"ventricular sax_ssfp compress/image___{study_uid}.nii.gz"
    masks_key  = f"ventricular sax_ssfp compress/masks___{study_uid}.nii.gz"

    # ----------------------------
    # Ensure all required objects exist
    # ----------------------------
    required = {
        "saxdf": saxdf_key,
        "image": image_key,
        "masks": masks_key,
    }

    for name, key in required.items():
        try:
            s3.head_object(Bucket=ARTIFACT_BUCKET, Key=key)
        except Exception:
            raise FileNotFoundError(f"Missing {name} artifact at s3://{ARTIFACT_BUCKET}/{key}")

    # ----------------------------
    # Download them
    # ----------------------------
    def _download(key):
        local_path = os.path.join(local_dir, os.path.basename(key))
        if not os.path.exists(local_path):
            # s3.download_file(ARTIFACT_BUCKET, key, local_path)
            with open(local_path, "wb") as f:
                s3.download_fileobj(ARTIFACT_BUCKET, key, f)

        return local_path

    if saxdf_bool:
        _download(saxdf_key)
    _download(image_key)
    _download(masks_key)

    return local_dir

def patch_pandas_compat():
    """
    Fix compatibility for unpickling DataFrames created with older pandas versions
    (where Int64Index, Float64Index, etc. lived in pandas.core.indexes.numeric).
    """
    import pandas as pd
    import sys
    import types

    # Create placeholder module: pandas.core.indexes.numeric
    numeric_mod_name = "pandas.core.indexes.numeric"
    if numeric_mod_name not in sys.modules:
        numeric_mod = types.ModuleType(numeric_mod_name)
        sys.modules[numeric_mod_name] = numeric_mod
    else:
        numeric_mod = sys.modules[numeric_mod_name]

    # Map old classes to new classes
    import pandas.core.indexes.base as base

    # These old classes existed in old pandas
    old_to_new = {
        "Int64Index": base.Index,
        "Float64Index": base.Index,
        "RangeIndex": pd.RangeIndex,
    }

    for old_name, new_cls in old_to_new.items():
        setattr(numeric_mod, old_name, new_cls)



def safe_unpickle(raw_bytes):
    """
    Robust unpickler that handles old pandas pickles gracefully.
    """
    try:
        return pickle.loads(raw_bytes)
    except Exception as e:
        if "pandas" in str(e) or "Int64Index" in str(e):
            patch_pandas_compat()
            return pickle.loads(raw_bytes)
        raise


def load_saxdf(local_dir, uid):
    """
    Loads SAXDF metadata from either:
        - saxdf___{uid}.pkl   (primary)
        - saxdf___{uid}.csv   (fallback)

    Always returns:
        pixelspacing, thickness, df
    """
    pkl_path = os.path.join(local_dir, f"saxdf___{uid}.pkl")

    # --- Case 1: PKL exists ---
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            raw_bytes = f.read()
        df = safe_unpickle(raw_bytes)

    else:
        raise FileNotFoundError(
            f"Missing saxdf___{uid}.pkl in {local_dir}"
        )

    if isinstance(df, list):
        df = pd.DataFrame(df)

    return df


def extract_raw_dcm_attrs(site, study_uid, series_uid, tmp_base="/tmp/rawdcm"):
    """
    Extract true PixelSpacing and SliceThickness directly from original DICOMs.

    Parameters
    ----------
    site : str
        Site prefix in ambra-data-lake (e.g., 'BCH', 'CHOP', etc.)
    study_uid : str
        Study Instance UID
    series_uid : str
        Series Instance UID

    Returns
    -------
    (pixelspacing, thickness)
        pixelspacing : float or np.nan
        thickness : float or np.nan
    """
    AMBRA_BUCKET = "ambra-data-lake"
    # Construct S3 prefix
    prefix = f"{site}/{study_uid}/{series_uid}/"

    # Find SOP UIDs (any .dcm)
    resp = s3.list_objects_v2(
        Bucket=AMBRA_BUCKET,
        Prefix=prefix,
        MaxKeys=50
    )

    contents = resp.get("Contents", [])
    if not contents:
        raise FileNotFoundError(f"No DICOM files found under {prefix}")

    # Grab first .dcm
    sop_key = None
    for obj in contents:
        if obj["Key"].lower().endswith(".dcm"):
            sop_key = obj["Key"]
            break

    if sop_key is None:
        raise FileNotFoundError(f"No .dcm files found under {prefix}")

    # Local temp path
    os.makedirs(tmp_base, exist_ok=True)
    local_path = os.path.join(tmp_base, os.path.basename(sop_key))

    # Download exactly one DICOM
    s3.download_file(AMBRA_BUCKET, sop_key, local_path)

    # Read via pydicom
    try:
        dcm = pydicom.dcmread(local_path)
    except Exception as e:
        raise ValueError(f"Failed to read DICOM {sop_key}: {e}")

    # Extract attributes
    # -----------------------------------------
    # Thickness
    try:
        thickness = float(dcm.SpacingBetweenSlices)
    except:
        thickness = float(dcm.SliceThickness)

    # Pixel Spacing
    pixel_spacing = float(dcm.PixelSpacing[0])

    return pixel_spacing, thickness


def lookup_study_by_study_uid(study_uid):
    """
    Directly fetch the study metadata using study_uid as the primary key.
    Table structure:
      Key: study_uid
      Attributes: fid, study_date, ser_uids, ...
    """
    resp = study_meta_table.get_item(Key={"study-uid": study_uid})
    return resp.get("Item")



def get_patient_and_date(study_uid):
    item = lookup_study_by_study_uid(study_uid)
    if not item:
        return None, None, None

    patient = item.get("fid", None)
    study_date = item.get("study_date", None)
    series_uids = item.get("ser_uids", None)

    return patient, study_date, series_uids



def upload_roundel_results(study_uid, sax_series_uid, csv_path, mask_path, gif_path):
    prefix = f"roundel-results/{study_uid}/{sax_series_uid}/"

    s3.upload_file(csv_path, RESULTS_BUCKET, f"{prefix}edited_metrics.csv")
    s3.upload_file(mask_path, RESULTS_BUCKET, f"{prefix}final_mask.nii.gz")
    s3.upload_file(gif_path, RESULTS_BUCKET, f"{prefix}final_visualization.gif")

    return True


def to_decimal(obj):
    """Recursively convert any floats/numpy numbers into Decimal for DynamoDB."""

    # numpy scalar → python scalar
    if isinstance(obj, (np.float32, np.float64)):
        obj = float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        obj = int(obj)

    # float → Decimal
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None  # DynamoDB cannot store NaN/inf
        return Decimal(str(round(obj, 6)))

    # int → Decimal (safe)
    if isinstance(obj, int):
        return Decimal(str(obj))

    # list → recurse
    if isinstance(obj, list):
        return [to_decimal(x) for x in obj]

    # dict → recurse
    if isinstance(obj, dict):
        return {k: to_decimal(v) for k, v in obj.items()}

    # everything else is fine (str, None, bool)
    return obj


def ddb_update_status(series_uid, status, extra=None):
    item = {
        "sax_series_uid": series_uid,
        "status": status,
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        item.update(extra)

    table = dynamodb.Table(os.getenv("ROUNDL_SESSIONS_TABLE", "roundel-sessions"))
    table.put_item(Item=to_decimal(item))


def uncompress_sop_uids(sop_dict):
    """Expand compressed SOP UID lists into explicit lists."""
    out = {}
    for s_uid, val in sop_dict.items():
        if isinstance(val, dict) and "common_prefix" in val and "suffixes" in val:
            out[s_uid] = [val["common_prefix"] + s for s in val["suffixes"]]
        elif isinstance(val, list):
            out[s_uid] = val
        else:
            out[s_uid] = []
    return out


def get_series_description(study_uid, series_uid):
    """
    Retrieve the DICOM Series Description for a given (study_uid, series_uid)
    using the SOP-level metadata table.
    """

    # 1. Get study-level series → sop mapping
    resp = study_meta_table.get_item(Key={"study-uid": study_uid})
    study_item = resp.get("Item")

    if not study_item:
        raise ValueError(f"Study UID not found in metadata table: {study_uid}")

    ser_sop_dict_raw = study_item.get("ser_sop_uids")
    if not ser_sop_dict_raw:
        raise ValueError(f"No series/SOP mapping found for study {study_uid}")

    # ser_sop_uids may be string repr of dict -> eval safely
    if isinstance(ser_sop_dict_raw, str):
        import ast
        ser_sop_dict = ast.literal_eval(ser_sop_dict_raw)
    else:
        ser_sop_dict = ser_sop_dict_raw

    # Sometimes compressed → expand
    ser_sop_dict = uncompress_sop_uids(ser_sop_dict)

    # 2. SOP list for the target series
    sop_list = ser_sop_dict.get(series_uid)
    if not sop_list:
        raise ValueError(f"No SOPs found for series_uid {series_uid} in study {study_uid}")

    # 3. Pick a representative SOP uid
    sample_sop_uid = sop_list[0]

    # 4. Lookup the SOP metadata
    resp = sop_tbl.get_item(Key={"sop-uid": sample_sop_uid})
    sop_item = resp.get("Item")

    if not sop_item:
        raise ValueError(f"No SOP metadata for sop_uid {sample_sop_uid}")

    # 5. Extract the description
    description = sop_item.get("Series Description", "")
    return description

def extract_series_uid_from_dicom_path(path):
    # Normalize all slashes to "/"
    clean = path.replace("\\", "/")

    # Remove empty tokens (because you have triple slashes //)
    parts = [p for p in clean.split("/") if p]

    if len(parts) < 2:
        raise ValueError(f"Cannot extract series UID from path: {path}")

    # Second-to-last is always the SeriesInstanceUID
    return parts[-2]


def ingest_study(study_uid):
    """
    Pre-populate DDB with all metadata needed for Roundel:
      - fid
      - study_date
      - description
      - pixelspacing
      - thickness
      - study_uid
      - status = staged
    """
    # 1. download artifacts so saxdf becomes available
    local_dir = download_sax_artifacts(study_uid)

    # 2. pixelspacing & thickness from saxdf
    saxdf = load_saxdf(local_dir, study_uid)

    # 3. get FID + study_date from study-meta table
    patient, study_date, series_uids = get_patient_and_date(study_uid)

    # 4. Extracting series ID form dcm path (s3 reference)
    series_uid = extract_series_uid_from_dicom_path(saxdf['dicom'].iloc[0])

    # 5. extracting pixelspacing & thickness direct from dcm file
    pixelspacing, thickness = extract_raw_dcm_attrs(
        site=patient[:3],
        study_uid=study_uid,
        series_uid=series_uid
    )

    # 4. description from sop-instance table
    description = get_series_description(study_uid, series_uid)

    item = {
        "study_uid": study_uid,
        "fid": patient,
        "study_date": study_date,
        "description": description,
        "pixelspacing": pixelspacing,
        "thickness": thickness,
        "status": "staged"
    }

    table.put_item(Item=to_decimal(item))
    return item


def save_masks_and_metrics(
    study_uid,
    pixelspacing,
    thickness,
    dia_idx,
    sys_idx,
    edited_mask,
    final_mask_2d,
    edited_gif_path,
    full_gif_path,
    raw_edv,
    raw_esv,
    raw_mass,
    raw_ef,
    metrics,
    patient,
    study_date,
    description,
):
    """
    Saves masks + metrics for Roundel:
      - uploads mask NIfTI
      - uploads GIFs
      - uploads metrics CSV
      - writes DynamoDB item
    """

    # Unpack computed metrics
    edv, esv, sv, ef, mass = metrics

    # ----------------------------------------
    # 1. LOCAL TEMP SAVE
    # ----------------------------------------
    tmp_dir = "/tmp/roundel_save"
    os.makedirs(tmp_dir, exist_ok=True)

    mask_path = os.path.join(tmp_dir, f"masks___{study_uid}.nii.gz")
    metrics_path = os.path.join(tmp_dir, f"metrics___{study_uid}.csv")

    # Save NIfTI mask
    nib_img = nib.Nifti1Image(final_mask_2d.astype(np.uint8), np.eye(4))
    nib.save(nib_img, mask_path)

    # Save metrics CSV
    pd.DataFrame({
        "study_uid": [study_uid],
        "fid": [patient],
        "study_date": [study_date],
        "description": [description],
        "pixelspacing": [pixelspacing],
        "thickness": [thickness],
        "edv_frame": [dia_idx],
        "esv_frame": [sys_idx],
        "edv": [edv],
        "esv": [esv],
        "stroke_volume": [sv],
        "ejection_fraction": [ef],
        "mass": [mass],
        "raw_edv": [raw_edv],
        "raw_esv": [raw_esv],
        "raw_ef": [raw_ef],
        "raw_mass": [raw_mass],
        "num_slices": [edited_mask.shape[2]],
        "num_frames": [edited_mask.shape[3]],
        "timestamp": [datetime.utcnow().isoformat()],
    }).to_csv(metrics_path, index=False)

    # ----------------------------------------
    # 2. UPLOAD TO S3
    # ----------------------------------------
    def upload(local, key):
        s3.upload_file(local, RESULTS_BUCKET, key)

    upload(mask_path,      f"{ROUNDEL_RESULTS_PREFIX}masks___{study_uid}.nii.gz")
    upload(metrics_path,   f"{ROUNDEL_RESULTS_PREFIX}metrics___{study_uid}.csv")
    upload(edited_gif_path, f"{ROUNDEL_RESULTS_PREFIX}cropped_gif___{study_uid}.gif")
    upload(full_gif_path,   f"{ROUNDEL_RESULTS_PREFIX}full_gif___{study_uid}.gif")

    # ----------------------------------------
    # 3. DYNAMODB PUT
    # ----------------------------------------
    ddb_item = to_decimal({
        "study_uid": study_uid,
        "fid": patient,
        "study_date": study_date,
        "description": description,
        "pixelspacing": pixelspacing,
        "thickness": thickness,
        "metrics": {
            "edv": edv,
            "esv": esv,
            "stroke_volume": sv,
            "ejection_fraction": ef,
            "mass": mass,
            "raw_edv": raw_edv,
            "raw_esv": raw_esv,
            "raw_ef": raw_ef,
            "raw_mass": raw_mass,
            "edv_frame": dia_idx,
            "esv_frame": sys_idx,
        },
        "timestamp": datetime.utcnow().isoformat(),
        "status": "completed",
    })

    table.put_item(Item=ddb_item)

    return True


def skip_case_ddb(study_uid):
    """Mark the study as skipped in DynamoDB."""
    dynamodb = boto3.resource("dynamodb")
    table = dynamodb.Table(os.getenv("DYNAMO_TABLE", "roundel-sessions"))

    table.update_item(
        Key={"study_uid": study_uid},
        UpdateExpression="SET #s = :skipped",
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues={":skipped": "skipped"}
    )
