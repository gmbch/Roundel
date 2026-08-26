import os
import boto3
import pandas as pd
import pickle
import base64
import binascii
import nibabel as nib
import numpy as np
from decimal import Decimal
import pydicom
import io
import json
from datetime import datetime, timezone
from functools import lru_cache
from urllib.parse import quote

# ------------------------------------------------------
# AWS Setup
# ------------------------------------------------------
ARTIFACT_BUCKET = os.getenv("ROUNDL_ARTIFACT_BUCKET", "dcmlab-img-storage")
# ARTIFACT_PREFIX = os.getenv("ROUNDL_ARTIFACT_PREFIX", "ventricular sax_ssfp compress/")
SAXDF_PREFIX = os.getenv("ROUNDL_ARTIFACT_PREFIX", "ventricular sax_ssfp/")

# RESULTS_BUCKET = os.getenv("ROUNDL_RESULTS_BUCKET", "dcmpipe-streamlit")
STUDY_META_TABLE = os.getenv("ROUNDL_META_TABLE", "ambra-data-lake-study-meta-table")
# ROUNDL_TABLE = os.getenv("ROUNDL_SESSIONS_TABLE", "roundel-sessions")

"""LIVE"""
RESULTS_BUCKET = 'dcmpipe-streamlit'
ROUNDL_TABLE = 'roundel-sessions'
TST_VAR = 'PROD'
ROUNDEL_RESULTS_PREFIX = "roundel-results/"
# #

# """DEV"""
# RESULTS_BUCKET = 'dcmpipe-streamlit-test'
# ROUNDL_TABLE = 'roundel-sessions-test'
# TST_VAR = 'TST'
# ROUNDEL_RESULTS_PREFIX = "roundel-results-test/"


s3 = boto3.client("s3")
dynamodb = boto3.resource("dynamodb")
study_meta_table = dynamodb.Table(STUDY_META_TABLE)
sop_tbl = dynamodb.Table("ambra-data-lake-sop-meta-table")
table = dynamodb.Table(ROUNDL_TABLE)

AMBRA_LINK_SECRET_ID = os.getenv(
    "ROUNDL_AMBRA_LINK_SECRET_ID", "ambra_link_secret"
)


@lru_cache(maxsize=1)
def _load_ambra_link_secrets(secret_id=AMBRA_LINK_SECRET_ID):
    """Load Ambra link credentials from Secrets Manager without exposing them."""
    region = os.getenv("ROUNDL_AWS_REGION") or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
    client_kwargs = {"region_name": region} if region else {}
    secretsmanager = boto3.client("secretsmanager", **client_kwargs)
    response = secretsmanager.get_secret_value(SecretId=secret_id)
    secret_string = response.get("SecretString")
    if not secret_string:
        raise ValueError(f"Ambra link secret `{secret_id}` has no SecretString payload")
    values = json.loads(secret_string)
    if not isinstance(values, dict):
        raise ValueError(f"Ambra link secret `{secret_id}` must contain a JSON object")
    hex_key = values.get("ambra_hex_key") or values.get("hex_key")
    account_id = values.get("ambra_account_id") or values.get("account_id")
    if not hex_key or not account_id:
        raise ValueError(
            "Ambra link secret must contain ambra_hex_key and ambra_account_id"
        )
    return str(hex_key), str(account_id)


def build_ambra_link(study_uid, secret_id=AMBRA_LINK_SECRET_ID):
    """Build the encrypted Ambra study link for a study UID."""
    if study_uid is None or pd.isna(study_uid) or not str(study_uid).strip():
        return None

    secret_key, account_id = _load_ambra_link_secrets(secret_id)
    key_bytes = binascii.unhexlify(secret_key)
    if len(key_bytes) != 16:
        raise ValueError("Ambra AES key must be 16 bytes (32 hexadecimal characters)")
    iv_bytes = binascii.unhexlify("30303030303030303030303030303030")
    payload = json.dumps(
        {"filter.study_uid.equals": str(study_uid)},
        separators=(",", ":"),
    ).encode("utf-8")
    pad_len = 16 - (len(payload) % 16)
    padded = payload + bytes([pad_len]) * pad_len

    try:
        from Crypto.Cipher import AES

        encrypted = AES.new(key_bytes, AES.MODE_CBC, iv_bytes).encrypt(padded)
    except ImportError:
        try:
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        except ImportError as exc:
            raise RuntimeError(
                "AES support is unavailable. Install dependencies from "
                "requirements-ambra-link.txt (pycryptodome)."
            ) from exc

        encryptor = Cipher(algorithms.AES(key_bytes), modes.CBC(iv_bytes)).encryptor()
        encrypted = encryptor.update(padded) + encryptor.finalize()

    value = quote(base64.b64encode(encrypted).decode("utf-8"), safe="")
    return f"https://force.ambrahealth.com/api/v3/link/external?u={quote(account_id, safe='')}&v={value}"





INTERMEDIATE_PREFIX = os.getenv("ROUNDEL_INTERMEDIATE_PREFIX", "roundel-intermediate/")


def _json_default(o):
    if isinstance(o, Decimal):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")


def intermediate_prefix(study_uid: str) -> str:
    return f"{INTERMEDIATE_PREFIX}{study_uid}/"


def save_intermediate_session(
    study_uid,
    edited_mask,
    edv_esv_selected,
    patient=None,
    study_date=None,
    description=None,
    model_ind=None,
    panel_idx=None,
    runtime=None,
):
    prefix = intermediate_prefix(study_uid)

    config = {
        "study_uid": study_uid,
        "edv_esv_selected": edv_esv_selected,
        "patient": patient,
        "study_date": study_date,
        "description": description,
        "model_ind": model_ind,
        "panel_idx": panel_idx,
        "runtime": runtime,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }

    # Save config JSON
    s3.put_object(
        Bucket=RESULTS_BUCKET,
        Key=f"{prefix}config.json",
        Body=json.dumps(config, default=_json_default).encode("utf-8"),
        ContentType="application/json",
    )

    # Save edited mask as npy
    buf = io.BytesIO()
    np.save(buf, edited_mask.astype(np.uint8))
    buf.seek(0)

    s3.put_object(
        Bucket=RESULTS_BUCKET,
        Key=f"{prefix}edited_mask.npy",
        Body=buf.getvalue(),
        ContentType="application/octet-stream",
    )

    # Mark DDB case as resumable
    table.update_item(
        Key={"study_uid": study_uid},
        UpdateExpression="""
            SET checkpoint_exists = :true,
                checkpoint_status = :saved,
                checkpoint_s3_prefix = :prefix,
                checkpoint_last_updated = :ts,
                panel_idx = :panel_idx,
                runtime = :runtime
        """,
        ExpressionAttributeValues=to_decimal({
            ":true": True,
            ":saved": "saved",
            ":prefix": prefix,
            ":ts": datetime.now(timezone.utc).isoformat(),
            ":panel_idx": panel_idx,
            ":runtime": runtime,
        }),
    )

    return True


def load_intermediate_session(study_uid):
    prefix = intermediate_prefix(study_uid)

    try:
        config_obj = s3.get_object(
            Bucket=RESULTS_BUCKET,
            Key=f"{prefix}config.json",
        )
        mask_obj = s3.get_object(
            Bucket=RESULTS_BUCKET,
            Key=f"{prefix}edited_mask.npy",
        )
    except Exception:
        return None

    config = json.loads(config_obj["Body"].read().decode("utf-8"))

    mask_bytes = io.BytesIO(mask_obj["Body"].read())
    edited_mask = np.load(mask_bytes).astype(np.uint8)

    return {
        "config": config,
        "edited_mask": edited_mask,
    }


def delete_intermediate_session(study_uid):
    prefix = intermediate_prefix(study_uid)

    resp = s3.list_objects_v2(
        Bucket=RESULTS_BUCKET,
        Prefix=prefix,
    )

    objects = [{"Key": obj["Key"]} for obj in resp.get("Contents", [])]

    if objects:
        s3.delete_objects(
            Bucket=RESULTS_BUCKET,
            Delete={"Objects": objects},
        )

    table.update_item(
        Key={"study_uid": study_uid},
        UpdateExpression="""
            REMOVE checkpoint_exists,
               checkpoint_status,
               checkpoint_s3_prefix,
               checkpoint_last_updated,
               panel_idx,
               runtime
        """
    )

    return True


def fetch_staged_roundel_cases():
    """
    DynamoDB table should contain entries like:
      {
          status: "staged" or "4c-staged",
          study_uid: "...",
          site: "...",
          fid: "...",
          (maybe other tracking metadata)
      }
    """

    scan_kwargs = {
        "FilterExpression": "#s IN (:staged, :fourc_staged)",
        "ExpressionAttributeNames": {"#s": "status"},
        "ExpressionAttributeValues": {
            ":staged": "staged",
            ":fourc_staged": "4c-staged",
        },
    }
    items = []
    while True:
        resp = table.scan(**scan_kwargs)
        items.extend(resp.get("Items", []))
        if not resp.get("LastEvaluatedKey"):
            break
        scan_kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
    return items


def download_sax_artifacts(study_uid, saxdf_bool=True, model_ind='2D'):
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
    saxdf_key = f"ventricular sax_ssfp/saxdf___{study_uid}.pkl"
    image_key = f"ventricular sax_ssfp compress/image___{study_uid}.nii.gz"
    if model_ind == '2D':
        masks_key  = f"ventricular sax_ssfp compress/masks___{study_uid}.nii.gz"
    elif model_ind == '4D':
        masks_key = f"ventricular sax_ssfp 4d/masks___{study_uid}.nii.gz"

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
                print(f'DOWNLOAD MADE: {key}')

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



# def upload_roundel_results(study_uid, sax_series_uid, csv_path, mask_path, gif_path):
#     prefix = f"roundel-results/{study_uid}/{sax_series_uid}/"
#
#     s3.upload_file(csv_path, RESULTS_BUCKET, f"{prefix}edited_metrics.csv")
#     s3.upload_file(mask_path, RESULTS_BUCKET, f"{prefix}final_mask.nii.gz")
#     s3.upload_file(gif_path, RESULTS_BUCKET, f"{prefix}final_visualization.gif")
#
#     return True


def to_decimal(obj):
    """Recursively convert any floats/numpy numbers into Decimal for DynamoDB."""

    # numpy scalar → python scalar
    if isinstance(obj, (np.float32, np.float64)):
        obj = float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        obj = int(obj)

    # bool should remain bool
    if isinstance(obj, bool):
        return obj

    # float → Decimal
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return Decimal(str(round(obj, 6)))

    # int → Decimal
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
    try:
        # 1. Get study-level series → sop mapping
        resp = study_meta_table.get_item(Key={"study-uid": study_uid})
        study_item = resp.get("Item")

        if not study_item:
            raise ValueError(f"Study UID not found in metadata table: {study_uid}")

        ser_sop_dict_raw = study_item.get("ser_sop_uids")
        if not ser_sop_dict_raw:
            raise ValueError(f"No series/SOP mapping found for study {study_uid}")

        if 'SOP UIDS EXCEED MAX ALLOWED SIZE' in ser_sop_dict_raw:
            raise ValueError(f"SOP UIDS EXCEED MAX ALLOWED SIZE for study {study_uid}")

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
    except:
        return "NA"

def extract_series_uid_from_dicom_path(path):
    # Normalize all slashes to "/"
    clean = path.replace("\\", "/")

    # Remove empty tokens (because you have triple slashes //)
    parts = [p for p in clean.split("/") if p]

    if len(parts) < 2:
        raise ValueError(f"Cannot extract series UID from path: {path}")

    # Second-to-last is always the SeriesInstanceUID
    return parts[-2]


def ingest_study(study_uid, edv, esv, mass, sm_flags, comments, gif_url, model_ind):
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

    # try:
    #     # 4. description from sop-instance table
    #     description = get_series_description(study_uid, series_uid)
    # except:
    #     description = saxdf['seriesdescription'].iloc[0]

    description = saxdf['seriesdescription'].iloc[0]

    item = {
        "study_uid": study_uid,
        "fid": patient,
        "study_date": study_date,
        "description": description,
        "edv": edv,
        'esv': esv,
        'mass': mass,
        "pixelspacing": pixelspacing,
        "thickness": thickness,
        "sm_flags": sm_flags,
        "comments": comments,
        "gif_url": gif_url,
        "model_ind": model_ind,
        "status": "staged"
    }

    table.put_item(Item=to_decimal(item))
    print(f"UPD {TST_VAR}")
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
    model_ind,
    fourc_series_uid=None,
    fourc_description=None
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
        "model_ind": [model_ind],
        "fourc_series_uid": [fourc_series_uid or ""],
        "fourc_description": [fourc_description or ""],
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
        "model_ind": model_ind,
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
        "selected_4c_series_uid": fourc_series_uid or "",
        "selected_4c_description": fourc_description or "",
    })

    table.put_item(Item=ddb_item)

    return True


def skip_case_ddb(study_uid):
    """Mark the study as skipped in DynamoDB."""
    table.update_item(
        Key={"study_uid": study_uid},
        UpdateExpression="SET #s = :skipped",
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues={":skipped": "skipped"}
    )

def circle_case_ddb(study_uid):
    """Mark the study as skipped in DynamoDB."""
    table.update_item(
        Key={"study_uid": study_uid},
        UpdateExpression="SET #s = :circle",
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues={":circle": "circle"}
    )


