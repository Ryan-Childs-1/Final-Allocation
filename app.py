from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import allocation_feature_engineering as fe
from allocation_nn_core import NumpyMLP, make_meta_features

APP_DIR = Path(__file__).resolve().parent
AK_SITES = set(fe.AK_SITES)
TARGET_COL = fe.TARGET_COLUMN

st.set_page_config(page_title="Allocation Multiple Model", page_icon="📦", layout="wide")

# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def read_json(path: Path, default=None):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default if default is not None else {}
    return default if default is not None else {}


def read_csv_safe(path: Path) -> pd.DataFrame:
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        pass
    return pd.DataFrame()


def fmt_int(x):
    try:
        return f"{int(float(x)):,}"
    except Exception:
        return "—"


def fmt_pct(x):
    try:
        return f"{float(x):.2%}"
    except Exception:
        return "—"


def fmt_num(x, digits=3):
    try:
        return f"{float(x):,.{digits}f}"
    except Exception:
        return "—"


def numeric(values) -> np.ndarray:
    return pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(float)


def final_alloc_display(values) -> pd.Series:
    arr = np.rint(numeric(values)).astype(np.int64)
    return pd.Series(np.where(arr > 0, arr.astype(object), ""))


def find_target_column(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if fe.COLUMN_ALIASES.get(fe._alias_key(c), c) == TARGET_COL:
            return c
    return TARGET_COL if TARGET_COL in df.columns else None


# -----------------------------------------------------------------------------
# File/model loading
# -----------------------------------------------------------------------------

def _load_model_bytes(model_name: str) -> Optional[io.BytesIO | Path]:
    """Find direct npz or split parts for a model file."""
    direct = APP_DIR / model_name
    if direct.exists():
        return direct

    # Common flat split pattern: model.npz.part000
    parts = sorted(APP_DIR.glob(f"{model_name}.part*"))
    if parts:
        return io.BytesIO(b"".join(p.read_bytes() for p in parts))

    # Manifest-driven split pattern, if exported that way.
    manifest = read_json(APP_DIR / "model_part_manifest.json", default={})
    model_info = (manifest.get("models") or {}).get(model_name, {})
    part_list = model_info.get("parts") or model_info.get("part_files") or []
    if part_list:
        resolved = [APP_DIR / p for p in part_list]
        if all(p.exists() for p in resolved):
            return io.BytesIO(b"".join(p.read_bytes() for p in resolved))
    return None


def load_one_model(model_name: str) -> Optional[NumpyMLP]:
    loc = _load_model_bytes(model_name)
    if loc is None:
        return None
    try:
        return NumpyMLP.load(loc)
    except Exception:
        return None


@st.cache_resource(show_spinner="Loading model registry and available neural-network files...")
def load_bundle() -> Dict:
    registry = read_json(APP_DIR / "registry.json", default={})
    feature_config = read_json(APP_DIR / "feature_config.json", default={})
    model_files = registry.get("models", {})
    models = {key: load_one_model(file_name) for key, file_name in model_files.items()}
    return {
        "registry": registry,
        "feature_config": feature_config,
        "models": models,
        "model_files": model_files,
        "summary": read_json(APP_DIR / "model_summary.json", default={}),
        "training_history": read_json(APP_DIR / "training_history.json", default={}),
        "tuning_output": read_json(APP_DIR / "tuning_output.json", default={}),
        "filter_report_json": read_json(APP_DIR / "component_training_filter_report.json", default=[]),
    }


def required_models_for_prediction(bundle: Dict) -> list[str]:
    """Core prediction needs; specialists may be optional."""
    return [
        "allocate_classifier", "allocate_ranker", "allocate_auxiliary", "allocate_regressor",
        "review_pass1_classifier", "review_pass1_ranker",
        "review_classifier", "review_ranker", "review_auxiliary", "review_regressor",
    ]


def missing_required_models(bundle: Dict) -> list[str]:
    models = bundle.get("models", {})
    return [m for m in required_models_for_prediction(bundle) if models.get(m) is None]


# -----------------------------------------------------------------------------
# Workbook reading / cleaning
# -----------------------------------------------------------------------------

def read_uploaded(uploaded, sheet_name: Optional[str] = None) -> pd.DataFrame:
    name = uploaded.name.lower()
    data = uploaded.getvalue()
    if name.endswith(".csv"):
        return fe.normalize_columns(pd.read_csv(io.BytesIO(data), low_memory=False))
    if name.endswith(".xlsb"):
        xl = pd.ExcelFile(io.BytesIO(data), engine="pyxlsb")
        sheet = sheet_name or ("3.3 Working Table" if "3.3 Working Table" in xl.sheet_names else xl.sheet_names[0])
        raw = pd.read_excel(io.BytesIO(data), sheet_name=sheet, engine="pyxlsb", header=None)
        return fe.detect_header_and_table(raw)
    xl = pd.ExcelFile(io.BytesIO(data))
    sheet = sheet_name or ("3.3 Working Table" if "3.3 Working Table" in xl.sheet_names else xl.sheet_names[0])
    raw = pd.read_excel(io.BytesIO(data), sheet_name=sheet, header=None)
    return fe.detect_header_and_table(raw)


def clean_model_rows(df: pd.DataFrame, only_model_rows: bool = True) -> pd.DataFrame:
    work = fe.normalize_columns(df.copy()).dropna(how="all").reset_index(drop=True)
    # Drop repeated header rows inside exports.
    row_text = work.apply(lambda r: "|".join("" if pd.isna(v) else str(v) for v in r.to_numpy()), axis=1).str.upper()
    repeated_header = row_text.str.contains("FINAL ALLOC", na=False) & row_text.str.contains("FLAG", na=False)
    work = work.loc[~repeated_header].reset_index(drop=True)
    if only_model_rows:
        work = fe.ensure_columns(work, include_target=False)
        work = work.loc[fe.eligible_mask(work)].reset_index(drop=True)
    return work


# -----------------------------------------------------------------------------
# Prediction logic for Model 3 Strong NN
# -----------------------------------------------------------------------------

def predict_segment(prefix: str, X: np.ndarray, models: Dict, threshold: float) -> Dict:
    clf = models.get(f"{prefix}_classifier")
    ranker = models.get(f"{prefix}_ranker")
    aux = models.get(f"{prefix}_auxiliary")
    sizer = models.get(f"{prefix}_regressor")
    if any(m is None for m in [clf, ranker, aux, sizer]):
        raise RuntimeError(f"Missing one or more {prefix} models.")
    prob = clf.predict(X).reshape(-1)
    rank = ranker.predict(X).reshape(-1)
    aux_out = aux.predict(X)
    meta_x = make_meta_features(X, prob, rank, aux_out)
    flm_pred = np.maximum(sizer.predict(meta_x).reshape(-1), 0.0)
    return {"prob": prob, "rank": rank, "aux": aux_out, "flm_pred": flm_pred, "positive": prob >= threshold}


def predict_review_pass1(X: np.ndarray, models: Dict, threshold: float) -> Optional[Dict]:
    clf = models.get("review_pass1_classifier")
    ranker = models.get("review_pass1_ranker")
    if clf is None or ranker is None:
        return None
    prob = clf.predict(X).reshape(-1)
    rank = ranker.predict(X).reshape(-1)
    return {"prob": prob, "rank": rank, "positive": prob >= threshold}


def run_prediction(df: pd.DataFrame, bundle: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    missing = missing_required_models(bundle)
    if missing:
        raise RuntimeError("Prediction cannot run because these required model files are missing: " + ", ".join(missing))

    registry = bundle["registry"]
    models = bundle["models"]
    X, feature_cfg, canon = fe.build_feature_matrix(df, config=bundle.get("feature_config", {}), fit=False)
    canon = fe.ensure_columns(canon, include_target=False)

    out = pd.DataFrame(index=canon.index)
    out["model_path"] = "none"
    out["pred_prob"] = 0.0
    out["rank_score"] = 0.0
    out["pred_flms_raw"] = 0.0
    out["review_pass1_prob"] = 0.0
    out["review_pass1_rank_score"] = 0.0
    out["review_pass1_units"] = 0
    out["Predicted Final Alloc"] = 0

    alloc_m = fe.allocate_mask(canon)
    review_m = fe.review_mask(canon)
    site802_m = fe.site802_mask(canon)
    ak_alloc_m = fe.ak_mask(canon) & alloc_m
    flm = np.maximum(fe.numeric_series(canon, "FLM").to_numpy(float), 1.0)
    dc = np.maximum(fe.numeric_series(canon, "Dc Avail").to_numpy(float), 0.0)
    supply = fe.numeric_series(canon, "Supply").to_numpy(float)
    raw_units = np.zeros(len(canon), dtype=float)
    seg_cache = {}

    # Allocate main path
    if alloc_m.any():
        idx = np.where(alloc_m)[0]
        pa = predict_segment("allocate", X[alloc_m], models, registry.get("allocate_threshold", 0.475))
        seg_cache["allocate"] = (alloc_m, pa)
        raw_units[idx] = np.where(pa["positive"], pa["flm_pred"] * flm[idx], 0.0)
        out.loc[alloc_m, "model_path"] = "allocate_main_nn"
        out.loc[alloc_m, "pred_prob"] = pa["prob"]
        out.loc[alloc_m, "rank_score"] = pa["rank"]
        out.loc[alloc_m, "pred_flms_raw"] = pa["flm_pred"]

    # Review two-pass path
    if review_m.any():
        idx = np.where(review_m)[0]
        rp1 = predict_review_pass1(X[review_m], models, registry.get("review_pass1_threshold", 0.425))
        pass1_units = np.zeros(len(idx), dtype=float)
        pass1_pos = np.zeros(len(idx), dtype=bool)
        if rp1 is not None:
            zero_supply_available = (supply[idx] <= 0) & (dc[idx] > 0)
            pass1_pos = rp1["positive"] & zero_supply_available
            pass1_units = np.where(pass1_pos, np.minimum(flm[idx], dc[idx]), 0.0)
            out.loc[review_m, "review_pass1_prob"] = rp1["prob"]
            out.loc[review_m, "review_pass1_rank_score"] = rp1["rank"]
            out.loc[review_m, "review_pass1_units"] = np.rint(pass1_units).astype(int)
            out.loc[idx[pass1_pos], "model_path"] = "review_pass1_zero_supply_single_flm"

        pr = predict_segment("review", X[review_m], models, registry.get("review_threshold", 0.5))
        seg_cache["review"] = (review_m, pr)
        pass2_units = np.where(pr["positive"], pr["flm_pred"] * flm[idx], 0.0)
        combined = np.maximum(pass1_units, pass2_units)
        raw_units[idx] = combined
        out.loc[review_m, "pred_prob"] = pr["prob"]
        out.loc[review_m, "rank_score"] = pr["rank"]
        out.loc[review_m, "pred_flms_raw"] = pr["flm_pred"]
        pass2_pos = pass2_units > 0
        out.loc[idx[pass2_pos & ~pass1_pos], "model_path"] = "review_pass2_main_nn"
        out.loc[idx[pass2_pos & pass1_pos], "model_path"] = "review_pass1_plus_pass2_main_nn"
        out.loc[idx[~pass2_pos & ~pass1_pos], "model_path"] = "review_two_pass_none"

    # Specialist models are intentionally disabled.
    # AK and Site 802 rows remain in the base Allocate / Review paths above.
    # This keeps the app robust when specialist .npz files are deleted from the deployment folder.

    pred = fe.postprocess_units(raw_units, canon)
    out["Predicted Final Alloc"] = pred.astype(int)
    return canon, out


# -----------------------------------------------------------------------------
# Metrics / audit
# -----------------------------------------------------------------------------

def metrics_for(df: pd.DataFrame, pred_col: str = "Predicted Final Alloc", actual_col: str = TARGET_COL, name: str = "All") -> Dict:
    actual = numeric(df[actual_col]) if actual_col in df.columns else np.zeros(len(df))
    pred = numeric(df[pred_col]) if pred_col in df.columns else np.zeros(len(df))
    flm = np.maximum(numeric(df.get("FLM", pd.Series([1] * len(df)))), 1.0)
    dc = np.maximum(numeric(df.get("Dc Avail", pd.Series([0] * len(df)))), 0.0)
    err = pred - actual
    return {
        "Segment": name,
        "Rows": int(len(df)),
        "MAE Units": float(np.mean(np.abs(err))) if len(df) else 0,
        "RMSE Units": float(np.sqrt(np.mean(err ** 2))) if len(df) else 0,
        "Exact Rate": float(np.mean(pred == actual)) if len(df) else 0,
        "Within 1 FLM": float(np.mean(np.abs(err) <= flm)) if len(df) else 0,
        "False Positives": int(((pred > 0) & (actual <= 0)).sum()),
        "False Negatives": int(((pred <= 0) & (actual > 0)).sum()),
        "Pred Units": int(round(pred.sum())),
        "Actual Units": int(round(actual.sum())),
        "Unit Delta": int(round(pred.sum() - actual.sum())),
        "Negative Violations": int((pred < 0).sum()),
        "Over-DC Violations": int((pred > dc + 1e-9).sum()),
    }


def build_audit_metrics(canon: pd.DataFrame, pred: pd.DataFrame, actual_values) -> pd.DataFrame:
    d = canon.copy()
    d[TARGET_COL] = numeric(actual_values)
    d["Predicted Final Alloc"] = numeric(pred["Predicted Final Alloc"])
    d["model_path"] = pred["model_path"].values
    rows = [metrics_for(d, name="All")]
    masks = {
        "Allocate": fe.allocate_mask(d),
        "Review": fe.review_mask(d),
        "Review pass 1 candidates": fe.review_mask(d) & (fe.numeric_series(d, "Supply").to_numpy(float) <= 0) & (fe.numeric_series(d, "Dc Avail").to_numpy(float) > 0),
        "AK Allocate": fe.ak_mask(d) & fe.allocate_mask(d),
        "AK Review": fe.ak_mask(d) & fe.review_mask(d),
        "Site 802 Allocate": fe.site802_mask(d) & fe.allocate_mask(d),
        "Site 802 Review": fe.site802_mask(d) & fe.review_mask(d),
    }
    for name, mask in masks.items():
        if mask.any():
            rows.append(metrics_for(d.loc[mask], name=name))
    for path, g in d.groupby("model_path"):
        rows.append(metrics_for(g, name=f"model_path::{path}"))
    return pd.DataFrame(rows)


def row_audit_table(cleaned: pd.DataFrame, canon: pd.DataFrame, pred: pd.DataFrame, actual_values) -> pd.DataFrame:
    out = cleaned.reset_index(drop=True).copy()
    actual = np.rint(numeric(actual_values)).astype(int)
    pred_units = np.rint(numeric(pred["Predicted Final Alloc"])).astype(int)
    out["Actual Final Alloc"] = actual
    out["Predicted Final Alloc Audit"] = pred_units
    out["Absolute Error Units"] = np.abs(pred_units - actual)
    out["Signed Error Units"] = pred_units - actual
    out["Exact Match"] = pred_units == actual
    out["False Positive"] = (pred_units > 0) & (actual <= 0)
    out["False Negative"] = (pred_units <= 0) & (actual > 0)
    out["model_path"] = pred["model_path"].values
    out["pred_prob"] = pred["pred_prob"].values
    out["rank_score"] = pred["rank_score"].values
    out["pred_flms_raw"] = pred["pred_flms_raw"].values
    out["review_pass1_prob"] = pred["review_pass1_prob"].values
    out["review_pass1_units"] = pred["review_pass1_units"].values
    return out


# -----------------------------------------------------------------------------
# Report display helpers
# -----------------------------------------------------------------------------

def display_metric_table(df: pd.DataFrame):
    if df.empty:
        st.info("No data available.")
        return
    show = df.copy()
    for col in show.columns:
        lc = col.lower()
        if "rate" in lc or col in ["Exact Rate", "Within 1 FLM"]:
            show[col] = pd.to_numeric(show[col], errors="coerce").map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
        elif any(k in lc for k in ["mae", "rmse", "score", "corr", "lift"]):
            show[col] = pd.to_numeric(show[col], errors="coerce").map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    st.dataframe(show, use_container_width=True)


def file_inventory() -> pd.DataFrame:
    rows = []
    for p in sorted(APP_DIR.iterdir()):
        if p.is_file():
            rows.append({"file": p.name, "size_mb": round(p.stat().st_size / (1024 * 1024), 3)})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------

bundle = load_bundle()
registry = bundle["registry"]
models = bundle["models"]
model_files = bundle["model_files"]
summary = bundle["summary"]
missing_core = missing_required_models(bundle)
all_model_rows = []
disabled_specialists = {
    "site802_allocate_specialist",
    "site802_review_specialist",
    "ak_allocate_specialist",
    "ak_review_specialist",
}
for key, file_name in model_files.items():
    if key in disabled_specialists:
        status = "Disabled by design - base model used"
    else:
        status = "Loaded" if models.get(key) is not None else "Missing"
    all_model_rows.append({
        "model": key,
        "file": file_name,
        "status": status,
    })
model_status_df = pd.DataFrame(all_model_rows)

st.title("Allocation Multiple Model")
st.markdown(
    """
<div style="padding:1rem 1.15rem;border:1px solid rgba(128,128,128,.25);border-radius:.85rem;margin-bottom:1rem;">
  <b>Model 3 Strong Neural Network interface</b><br>
  Upload allocation workbooks, generate <code>Final Alloc.</code> predictions when model files are present, and review training/testing diagnostics from the packaged outputs.
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Model status")
    loaded_count = int((model_status_df["status"] == "Loaded").sum()) if not model_status_df.empty else 0
    total_count = len(model_status_df)
    st.metric("Loaded models", f"{loaded_count}/{total_count}")
    if missing_core:
        st.warning("Prediction is disabled until required base model files are added.")
    else:
        st.success("Base Allocate / Review models loaded. AK and Site 802 specialists are disabled by design.")
    st.divider()
    st.caption("Version")
    st.write(registry.get("version", "Unknown"))

predict_tab, audit_tab, overview_tab, performance_tab, features_tab, files_tab = st.tabs([
    "Predict", "Audit", "Model overview", "Performance", "Features", "Files"
])

with predict_tab:
    st.subheader("Predict Final Alloc.")
    if missing_core:
        st.info("Add the trained `.npz` model files listed on the Files tab to enable prediction. The app can still display reports and package diagnostics without the model files.")
        st.dataframe(model_status_df, use_container_width=True)
    only_model_rows = st.checkbox("Only process Allocate and Review rows", value=True, key="predict_only_model")
    up = st.file_uploader("Upload allocation workbook or CSV", type=["xlsb", "xlsx", "xlsm", "xls", "csv"], key="predict_upload")
    if up is not None:
        try:
            raw = read_uploaded(up)
            cleaned = clean_model_rows(raw, only_model_rows)
            if missing_core:
                st.warning("File was read successfully, but prediction cannot run because model files are missing.")
                st.metric("Rows ready for prediction", f"{len(cleaned):,}")
                st.dataframe(cleaned.head(200), use_container_width=True)
            else:
                canon, pred = run_prediction(cleaned, bundle)
                output = cleaned.reset_index(drop=True).copy()
                target_col = find_target_column(output) or TARGET_COL
                output[target_col] = final_alloc_display(pred["Predicted Final Alloc"]).values
                pred_units = numeric(pred["Predicted Final Alloc"])
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Rows", f"{len(output):,}")
                c2.metric("Nonzero allocations", f"{int((pred_units > 0).sum()):,}")
                c3.metric("Predicted units", f"{int(pred_units.sum()):,}")
                c4.metric("Model paths used", f"{pred['model_path'].nunique():,}")
                st.success("Prediction completed.")
                filter_choice = st.selectbox("Spot-check rows", ["All", "Nonzero predictions", "Allocate", "Review", "AK", "Site 802", "Review pass 1", "Largest predicted units"])
                mask = np.ones(len(output), dtype=bool)
                if filter_choice == "Nonzero predictions":
                    mask = pred_units > 0
                elif filter_choice == "Allocate":
                    mask = fe.allocate_mask(canon)
                elif filter_choice == "Review":
                    mask = fe.review_mask(canon)
                elif filter_choice == "AK":
                    mask = fe.ak_mask(canon)
                elif filter_choice == "Site 802":
                    mask = fe.site802_mask(canon)
                elif filter_choice == "Review pass 1":
                    mask = pred["review_pass1_units"].to_numpy() > 0
                preview = output.loc[mask].copy()
                if filter_choice == "Largest predicted units":
                    preview = output.assign(_pred_units=pred_units).sort_values("_pred_units", ascending=False).drop(columns=["_pred_units"])
                st.dataframe(preview.head(1000), use_container_width=True)
                audit_download = pd.concat([canon.reset_index(drop=True), pred.reset_index(drop=True)], axis=1)
                st.download_button("Download filled CSV", output.to_csv(index=False).encode("utf-8"), file_name="allocation_multiple_model_filled_output.csv", mime="text/csv", key="download_prediction_output")
                st.download_button("Download prediction audit CSV", audit_download.to_csv(index=False).encode("utf-8"), file_name="allocation_multiple_model_prediction_audit.csv", mime="text/csv", key="download_prediction_audit")
        except Exception as e:
            st.error("Prediction failed.")
            st.exception(e)

with audit_tab:
    st.subheader("Audit uploaded file")
    st.write("Upload a workbook that already has `Final Alloc.` populated to compare model predictions against actual allocations.")
    only_model_rows_audit = st.checkbox("Only audit Allocate and Review rows", value=True, key="audit_only_model")
    audit_up = st.file_uploader("Upload file for audit", type=["xlsb", "xlsx", "xlsm", "xls", "csv"], key="audit_upload")
    if audit_up is not None:
        try:
            raw = read_uploaded(audit_up)
            cleaned = clean_model_rows(raw, only_model_rows_audit)
            target_col = find_target_column(cleaned)
            if target_col is None:
                st.warning("No `Final Alloc.` column was detected.")
            elif missing_core:
                st.warning("Model files are missing, so the app cannot calculate live audit predictions. Use the Performance tab for packaged test results.")
                st.dataframe(cleaned.head(200), use_container_width=True)
            else:
                canon, pred = run_prediction(cleaned, bundle)
                metrics = build_audit_metrics(canon, pred, cleaned[target_col])
                row_audit = row_audit_table(cleaned, canon, pred, cleaned[target_col])
                all_row = metrics.iloc[0]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Rows audited", fmt_int(all_row["Rows"]))
                c2.metric("MAE units", fmt_num(all_row["MAE Units"]))
                c3.metric("Exact rate", fmt_pct(all_row["Exact Rate"]))
                c4.metric("Unit delta", fmt_int(all_row["Unit Delta"]))
                display_metric_table(metrics)
                st.markdown("#### Row-level audit")
                err_filter = st.selectbox("Filter", ["All", "Errors only", "False positives", "False negatives", "AK", "Site 802", "Review pass 1"])
                mask = np.ones(len(row_audit), dtype=bool)
                if err_filter == "Errors only":
                    mask = row_audit["Absolute Error Units"].to_numpy() > 0
                elif err_filter == "False positives":
                    mask = row_audit["False Positive"].to_numpy()
                elif err_filter == "False negatives":
                    mask = row_audit["False Negative"].to_numpy()
                elif err_filter == "AK":
                    mask = fe.ak_mask(canon)
                elif err_filter == "Site 802":
                    mask = fe.site802_mask(canon)
                elif err_filter == "Review pass 1":
                    mask = row_audit["review_pass1_units"].to_numpy() > 0
                st.dataframe(row_audit.loc[mask].head(1000), use_container_width=True)
                st.download_button("Download audit metrics CSV", metrics.to_csv(index=False).encode("utf-8"), file_name="uploaded_file_audit_metrics.csv", mime="text/csv", key="download_audit_metrics")
                st.download_button("Download row-level audit CSV", row_audit.to_csv(index=False).encode("utf-8"), file_name="uploaded_file_row_audit.csv", mime="text/csv", key="download_row_audit")
        except Exception as e:
            st.error("Audit failed.")
            st.exception(e)

with overview_tab:
    st.subheader("Model overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training rows", fmt_int(registry.get("rows", summary.get("rows"))))
    c2.metric("Features", fmt_int(registry.get("feature_count", summary.get("features"))))
    c3.metric("Allocate rows", fmt_int(summary.get("allocate_rows", "")))
    c4.metric("Review rows", fmt_int(summary.get("review_rows", "")))

    st.markdown("#### Architecture")
    arch = registry.get("architecture") or summary.get("architecture") or {}
    if arch:
        st.json(arch)
    else:
        st.info("No architecture metadata found.")

    st.markdown("#### Model status")
    st.dataframe(model_status_df, use_container_width=True)

    st.markdown("#### Component training filters")
    comp_filter = read_csv_safe(APP_DIR / "component_training_filter_report.csv")
    if not comp_filter.empty:
        st.dataframe(comp_filter, use_container_width=True)
    else:
        st.info("component_training_filter_report.csv not found.")

    with st.expander("Registry JSON", expanded=False):
        st.json(registry)

with performance_tab:
    st.subheader("Packaged testing results")
    overall = read_csv_safe(APP_DIR / "overall_segment_metrics.csv")
    component = read_csv_safe(APP_DIR / "component_model_metrics.csv")
    grouped = read_csv_safe(APP_DIR / "grouped_metrics_all.csv")
    ranker = read_csv_safe(APP_DIR / "ranker_topk_lift.csv")
    largest = read_csv_safe(APP_DIR / "largest_errors_top500.csv")
    test_report = APP_DIR / "TEST_REPORT.md"

    if not overall.empty:
        st.markdown("#### Segment metrics")
        display_metric_table(overall)
    if not component.empty:
        st.markdown("#### Component model metrics")
        display_metric_table(component)
    if not ranker.empty:
        st.markdown("#### Ranker Top-K lift")
        display_metric_table(ranker)
    with st.expander("Grouped metrics", expanded=False):
        if not grouped.empty:
            st.dataframe(grouped, use_container_width=True)
    with st.expander("Largest errors", expanded=False):
        if not largest.empty:
            st.dataframe(largest, use_container_width=True)
    if test_report.exists():
        with st.expander("Full TEST_REPORT.md", expanded=False):
            st.markdown(test_report.read_text(encoding="utf-8"))
    if (APP_DIR / "test_results_bundle.zip").exists():
        st.download_button("Download packaged test results", (APP_DIR / "test_results_bundle.zip").read_bytes(), file_name="test_results_bundle.zip", mime="application/zip", key="download_test_results_bundle")

with features_tab:
    st.subheader("Feature system")
    feature_catalog = read_csv_safe(APP_DIR / "feature_catalog.csv")
    feature_config = read_json(APP_DIR / "feature_config.json", default={})
    c1, c2, c3 = st.columns(3)
    c1.metric("Approved columns", fmt_int(len(feature_config.get("approved_columns", []))))
    c2.metric("Final features", fmt_int(len(feature_config.get("final_feature_names", []))))
    c3.metric("Extra engineered features", fmt_int(len(feature_config.get("extra_feature_names", []))))

    st.markdown("#### Approved worksheet columns")
    st.write(", ".join(feature_config.get("approved_columns", fe.APPROVED_COLUMNS)))

    st.markdown("#### Feature families")
    st.markdown(
        """
- **Original worksheet signals:** demand, supply, rank, FLM, cost, projected demand, and allocation recommendation.
- **Demand agreement:** measures whether L30, D30, D60, LW, TTM, and projected demand agree that allocation is needed.
- **Supply pressure:** compares need signals against current supply and available DC units.
- **Pack-size logic:** estimates whether one FLM, two FLMs, or the recommendation would over-supply the store.
- **Class/line/site context:** compares a row against peers in the same class/line and site groups.
- **State and AK routing:** identifies AK stores by `State == AK` or the approved AK site list.
- **Site 802 routing:** identifies Site 802 by site value or blank-rank behavior when present.
- **Two-pass Review features:** separates zero-supply single-FLM review rescue from normal Review allocation.
"""
    )
    if not feature_catalog.empty:
        st.markdown("#### Feature catalog")
        st.dataframe(feature_catalog, use_container_width=True)
        st.download_button("Download feature catalog", feature_catalog.to_csv(index=False).encode("utf-8"), file_name="feature_catalog.csv", mime="text/csv", key="download_feature_catalog")
    with st.expander("Feature configuration JSON", expanded=False):
        st.json(feature_config)

with files_tab:
    st.subheader("Files and deployment readiness")
    st.dataframe(file_inventory(), use_container_width=True)
    st.markdown("#### Required model files")
    st.dataframe(model_status_df, use_container_width=True)
    if missing_core:
        st.warning("Missing required prediction models: " + ", ".join(missing_core))
    else:
        st.success("All required prediction models are loaded.")
    st.markdown("#### Notes")
    st.write("Specialist model files are not required. AK and Site 802 rows are intentionally routed through the base Allocate / Review neural models.")
