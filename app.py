from __future__ import annotations
import os, io, json, zipfile, math
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

import allocation_split_numpy_core as core
import allocation_v35_pruned_enhancements as enh35
import allocation_v36_context_features as enh36
import allocation_v37_competition_features as enh37

APP_DIR = Path(__file__).resolve().parent
ART = APP_DIR
REPORTS = APP_DIR

AK_SITES = {"248", "159", "212", "145", "121"}

st.set_page_config(
    page_title="Allocation Multiple Model",
    layout="wide",
)
st.title("Allocation Multiple Model")
st.markdown(
    """
    <div style="padding: 0.85rem 1rem; border: 1px solid rgba(128,128,128,0.25); border-radius: 0.75rem; margin-bottom: 1rem;">
        <strong>Upload an allocation workbook, generate Final Alloc. predictions, and audit results when an existing Final Alloc. column is present.</strong>
        <div style="opacity: 0.75; margin-top: 0.25rem;">Use the tabs below to predict, audit, review model details, inspect features, or confirm packaged files.</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Loading helpers
# -----------------------------------------------------------------------------

def _read_json(path: Path, default=None):
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {} if default is None else default


def _load_npz_from_flat_or_parts(name: str):
    """Load a model stored either as name.npz or flat split files name.npz.part000..."""
    direct = ART / name
    if direct.exists():
        return np.load(direct, allow_pickle=True)
    part_paths = sorted(ART.glob(f"{name}.part*"))
    if not part_paths:
        raise FileNotFoundError(f"Could not find {name} or split parts {name}.part000...")
    data = b"".join(p.read_bytes() for p in part_paths)
    return np.load(io.BytesIO(data), allow_pickle=True)


def _model_from_compact(z, role: str):
    meta = json.loads(str(z[f"{role}__meta"].item()))
    model = core.NumpyMLP(
        meta["input_dim"],
        meta["output_dim"],
        tuple(meta["hidden"]),
        meta["task"],
        dropout=meta.get("dropout", 0.0),
    )
    n = len(meta["hidden"]) + 1
    model.W = [z[f"{role}__W{i}"].astype(np.float32) for i in range(n)]
    model.b = [z[f"{role}__b{i}"].astype(np.float32) for i in range(n)]
    model.mW = [np.zeros_like(w) for w in model.W]
    model.vW = [np.zeros_like(w) for w in model.W]
    model.mb = [np.zeros_like(b) for b in model.b]
    model.vb = [np.zeros_like(b) for b in model.b]
    return model


@st.cache_resource(show_spinner="Loading base Allocate and Review model bundle...")
def load_bundle():
    meta = _read_json(ART / "model_config.json")
    fc = meta.get("feature_config", {})
    feat_cfg = core.FeatureConfig(
        hash_dim_class=fc.get("hash_dim_class", 96),
        hash_dim_line=fc.get("hash_dim_line", 128),
        hash_dim_site=fc.get("hash_dim_site", 96),
        hash_dim_rank=fc.get("hash_dim_rank", 8),
        hash_dim_flag=fc.get("hash_dim_flag", 8),
        hash_dim_dc_bucket=fc.get("hash_dim_dc_bucket", 8),
        hash_dim_raw_dc_bucket=fc.get("hash_dim_raw_dc_bucket", 12),
        numeric_mean=fc.get("numeric_mean"),
        numeric_std=fc.get("numeric_std"),
        feature_names=fc.get("feature_names"),
    )
    models = {}
    for seg in ["allocate", "review"]:
        z = _load_npz_from_flat_or_parts(f"{seg}_model.npz")
        clf = _model_from_compact(z, "classifier")
        reg = _model_from_compact(z, "regressor")
        models[seg] = {
            "classifier": clf,
            "regressor": reg,
            "classifiers": [clf],
            "regressors": [reg],
            "residual": None,
        }
    params35 = enh35.load_v35_params(str(ART / "v35_pruned_params.json"))
    params36 = enh36.load_v36_params(str(ART / "v36_context_params.json"))
    params37 = enh37.load_v37_params(str(ART / "v37_competition_params.json"))
    bundle = {
        "meta": meta,
        "feature_config": feat_cfg,
        "models": models,
        "site802_model": None,
        "ak_specialist_model": None,
        "specialists_disabled": True,
    }
    return bundle, params35, params36, params37


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------

def safe_cell_to_str(x):
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)


def clean_site_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(".0", "", regex=False).str.strip()


def find_header_row(df):
    wanted = set(core.ALLOWED_FEATURES + [core.TARGET_COL])
    best_i, best_hits = 0, -1
    for i in range(min(len(df), 90)):
        hits = sum(
            1
            for v in df.iloc[i].tolist()
            if core.CANONICAL_ALIASES.get(core._norm_name(v)) in wanted
        )
        if hits > best_hits:
            best_i, best_hits = i, hits
    return best_i if best_hits >= 8 else 0


def read_upload(uploaded, sheet_name=None):
    name = uploaded.name.lower()
    data = uploaded.getvalue()
    if name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(data), low_memory=False)
    if name.endswith(".xlsb"):
        xl = pd.ExcelFile(io.BytesIO(data), engine="pyxlsb")
        sheet = sheet_name or ("3.3 Working Table" if "3.3 Working Table" in xl.sheet_names else xl.sheet_names[0])
        preview = pd.read_excel(io.BytesIO(data), sheet_name=sheet, engine="pyxlsb", header=None, nrows=90)
        hdr = find_header_row(preview)
        return pd.read_excel(io.BytesIO(data), sheet_name=sheet, engine="pyxlsb", header=hdr)
    xl = pd.ExcelFile(io.BytesIO(data))
    sheet = sheet_name or ("3.3 Working Table" if "3.3 Working Table" in xl.sheet_names else xl.sheet_names[0])
    preview = pd.read_excel(io.BytesIO(data), sheet_name=sheet, header=None, nrows=90)
    hdr = find_header_row(preview)
    return pd.read_excel(io.BytesIO(data), sheet_name=sheet, header=hdr)


def drop_unnecessary_rows(df, drop_non_model=True):
    work = df.dropna(how="all").copy()
    row_text = work.apply(lambda r: "|".join(safe_cell_to_str(v) for v in r.to_numpy()), axis=1).str.upper()
    repeated = row_text.str.contains("FINAL ALLOC", na=False) & row_text.str.contains("ALLOC", na=False) & row_text.str.contains("FLAG", na=False)
    work = work.loc[~repeated].reset_index(drop=True)
    if drop_non_model:
        try:
            canon = core.canonicalize_columns(work, target_required=False)
            flags = canon["Flag"].astype(str).str.upper()
            keep = (flags.str.contains("ALLOC", na=False) & ~flags.str.contains("NO", na=False)) | flags.str.contains("REVIEW", na=False)
            work = work.loc[keep.to_numpy()].reset_index(drop=True)
        except Exception:
            pass
    return work


def find_target_column(df: pd.DataFrame):
    for c in df.columns:
        if core.CANONICAL_ALIASES.get(core._norm_name(c)) == core.TARGET_COL:
            return c
    return None


def int_or_blank_series(values) -> pd.Series:
    """Return display-safe integer Final Alloc values: integer for >0, blank otherwise."""
    nums = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    nums = np.rint(nums.to_numpy(dtype=float)).astype(np.int64)
    return pd.Series(np.where(nums > 0, nums.astype(object), ""))


def numeric_units(values) -> np.ndarray:
    return pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)


def predict_allocation(cleaned: pd.DataFrame):
    bundle, params35, params36, params37 = load_bundle()
    base = core.predict_dataframe(cleaned, bundle, target_required=False)
    canon = core.canonicalize_columns(cleaned, target_required=False)
    audit = enh35.apply_v35_pruned_no_residual(canon, base, params35)
    audit = enh36.apply_v36_context_enhanced(canon, audit, params36)
    audit = enh37.apply_v37_competition_aware(canon, audit, params37)
    # Specialist models are intentionally disabled. AK and Site 802 rows use the base
    # Allocate / Review model outputs, just like all other sites.
    audit = audit.copy()
    audit["Predicted Final Alloc"] = int_or_blank_series(audit["Predicted Final Alloc"]).values
    return canon, audit


# -----------------------------------------------------------------------------
# Metrics and audit helpers
# -----------------------------------------------------------------------------

def build_segment_masks(canon: pd.DataFrame, audit: pd.DataFrame):
    flags = canon["Flag"].astype(str).str.upper()
    sites = clean_site_series(canon["Site"])
    pred = numeric_units(audit["Predicted Final Alloc"])
    masks = {
        "All rows": np.ones(len(canon), dtype=bool),
        "Allocate rows": (flags.str.contains("ALLOC", na=False) & ~flags.str.contains("NO", na=False)).to_numpy(),
        "Review rows": flags.str.contains("REVIEW", na=False).to_numpy(),
        "Site 802 rows": sites.eq("802").to_numpy(),
        "AK store rows": sites.isin(AK_SITES).to_numpy(),
        "Nonzero predictions": pred > 0,
        "Blank / zero predictions": pred <= 0,
    }
    return masks


def metric_row(name: str, mask: np.ndarray, pred: np.ndarray, actual: np.ndarray, flm: np.ndarray, dc: np.ndarray):
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() == 0:
        return None
    p = pred[mask]
    a = actual[mask]
    f = np.maximum(flm[mask], 1.0)
    d = dc[mask]
    err = p - a
    abs_err = np.abs(err)
    return {
        "Segment / model path": name,
        "Rows": int(mask.sum()),
        "MAE Units": float(np.mean(abs_err)),
        "RMSE Units": float(np.sqrt(np.mean(err ** 2))),
        "Exact Rate": float(np.mean(p == a)),
        "Within 1 FLM": float(np.mean(abs_err <= f)),
        "False Positives": int(((p > 0) & (a <= 0)).sum()),
        "False Negatives": int(((p <= 0) & (a > 0)).sum()),
        "Pred Units": int(round(float(p.sum()))),
        "Actual Units": int(round(float(a.sum()))),
        "Unit Delta": int(round(float(p.sum() - a.sum()))),
        "Negative Violations": int((p < 0).sum()),
        "Over-DC Violations": int((p - d > 1e-9).sum()),
    }


def compute_audit_metrics(canon: pd.DataFrame, audit: pd.DataFrame, actual_values) -> pd.DataFrame:
    pred = numeric_units(audit["Predicted Final Alloc"])
    actual = numeric_units(actual_values)
    flm = numeric_units(canon["FLM"])
    dc = numeric_units(canon["Dc Avail"])
    rows = []
    for name, mask in build_segment_masks(canon, audit).items():
        row = metric_row(name, mask, pred, actual, flm, dc)
        if row is not None:
            rows.append(row)
    return pd.DataFrame(rows)


def build_row_audit(cleaned: pd.DataFrame, canon: pd.DataFrame, audit: pd.DataFrame, actual_values) -> pd.DataFrame:
    out = cleaned.copy()
    pred = numeric_units(audit["Predicted Final Alloc"])
    actual = numeric_units(actual_values)
    out["Actual Final Alloc"] = np.rint(actual).astype(int)
    out["Predicted Final Alloc Audit"] = np.rint(pred).astype(int)
    out["Absolute Error Units"] = np.abs(out["Predicted Final Alloc Audit"] - out["Actual Final Alloc"])
    out["Signed Error Units"] = out["Predicted Final Alloc Audit"] - out["Actual Final Alloc"]
    out["Exact Match"] = out["Absolute Error Units"].eq(0)
    out["False Positive"] = (out["Predicted Final Alloc Audit"] > 0) & (out["Actual Final Alloc"] <= 0)
    out["False Negative"] = (out["Predicted Final Alloc Audit"] <= 0) & (out["Actual Final Alloc"] > 0)
    out["Model Segment"] = np.where(canon["Flag"].astype(str).str.upper().str.contains("REVIEW", na=False), "Review", "Allocate")
    out["Site 802 Specialist Applied"] = 0
    out["AK Specialist Applied"] = 0
    out["Specialist Routing"] = "Disabled - base Allocate/Review models used"
    out["Allocation Confidence"] = audit.get("Allocation Confidence", pd.Series([np.nan] * len(audit))).values
    out["Raw Predicted FLMs"] = audit.get("Raw Predicted FLMs", pd.Series([np.nan] * len(audit))).values
    return out


# -----------------------------------------------------------------------------
# Feature deep-dive helpers
# -----------------------------------------------------------------------------

def _load_feature_matrix():
    p = REPORTS / "feature_decision_matrix.csv"
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def overview_stats(bundle):
    meta = bundle.get("meta", {})
    summary = _read_json(ART / "model_summary.json")
    train_rows = summary.get("rows") or meta.get("rows") or meta.get("training_rows")
    if not train_rows:
        # Some older compact exports did not store train rows. Use the packaged v3.9 evaluation row count
        # as a nonzero visibility fallback rather than displaying an incorrect zero.
        try:
            sm = pd.read_csv(REPORTS / "v39_summary_metrics_weighted.csv")
            row = sm[(sm["model"].astype(str).str.contains("v3_9", na=False)) & (sm["segment"].eq("All"))].head(1)
            if not row.empty:
                train_rows = f"{int(row.iloc[0]['rows']):,} audited rows"
        except Exception:
            train_rows = "Not stored in artifact"
    else:
        train_rows = f"{int(train_rows):,}"

    features = summary.get("features") or meta.get("features") or meta.get("feature_count")
    if not features:
        try:
            features = int(bundle["models"]["allocate"]["classifier"].input_dim)
        except Exception:
            fc = meta.get("feature_config", {})
            features = len(fc.get("feature_names", []) or [])
    return train_rows, features


def show_feature_deep_dive(bundle):
    meta = bundle.get("meta", {})
    feature_df = _load_feature_matrix()
    st.subheader("Feature deep dive and feature-importance review")
    st.markdown(
        """
This page replaces the old smoke-test report. It is focused on **what the model uses**, **why those features matter**, and **how the feature-pruned v3.9 stack decides allocations**.

The v3.9 model is intentionally built around the original worksheet fields first. The engineered features do not introduce outside data; they transform the approved columns into signals that are easier for the Allocate model, Review model, the base Allocate and Review models to use.
"""
    )

    c1, c2, c3, c4 = st.columns(4)
    train_rows, feature_count = overview_stats(bundle)
    c1.metric("Packaged model inputs", f"{int(feature_count):,}" if isinstance(feature_count, (int, float)) else str(feature_count))
    c2.metric("Feature decisions", f"{len(feature_df):,}" if not feature_df.empty else "—")
    c3.metric("Kept features", f"{int((feature_df.get('decision', pd.Series(dtype=str)).astype(str).str.upper() == 'KEEP').sum()):,}" if not feature_df.empty else "—")
    c4.metric("Removed features", f"{int((feature_df.get('decision', pd.Series(dtype=str)).astype(str).str.upper() == 'REMOVE').sum()):,}" if not feature_df.empty else "—")

    st.markdown("### 1. How features flow through the model")
    st.markdown(
        """
| Model layer | Main feature use | Why it matters |
|---|---|---|
| **Allocate classifier** | `Flag`, `Alloc. Rec.`, `Proj. Demand`, `Supply`, `Dc Avail`, shortage/pressure signals, class-line identity | Decides whether a row deserves any allocation at all. |
| **Allocate FLM regressor** | `FLM`, `MIL`, `Dc Avail`, `Alloc. Rec.`, `Proj. Demand`, post-allocation supply-risk features | Estimates how many FLMs should be allocated after the row has passed the classifier. |
| **Review classifier / ranker** | demand agreement, shortage FLMs, rank score, class-line competition, DC pressure | Review rows are ranking-like: the model prioritizes stores until DC inventory is exhausted. |
| **Base Allocate model** | Allocate rows, recommendation trust, demand pressure, FLM sizing | All Allocate rows, including AK and Site 802, are handled by this base path. |
| **Base Review model** | Review rows, demand agreement, shortage FLMs, rank score, class-line competition | All Review rows, including AK and Site 802, are handled by this base path. |
"""
    )

    st.markdown("### 2. Feature families that are most important")
    st.markdown(
        """
**Original worksheet fields are the foundation.** The model still sees the fields you manually rely on: `Alloc. Rec.`, `Proj. Demand`, `Supply`, `Dc Avail`, `FLM`, `MIL`, `Rank`, `L30`, `D30`, `D60`, `LW`, `TTM`, `Class Name`, `Line Name`, and `Site`.

The highest-value engineered families are:

1. **Demand agreement features** — count how many demand signals agree inventory is needed.
2. **Shortage and pressure features** — compare demand/projection against current supply.
3. **Pack-size features** — show how meaningful one FLM is relative to demand.
4. **Post-allocation risk features** — estimate whether following the recommendation would over-supply the store.
5. **Class-line competition features** — compare a row to peer rows in the same class/line group.
6. **Raw Dc Avail buckets** — keep your previously requested 25/50/100/300/600/1000/2000/2000+ DC bands visible.
7. **Site / AK indicators** — still visible for auditing, but specialist override models are disabled by design.
"""
    )

    if not feature_df.empty:
        st.markdown("### 3. Feature pruning decision matrix")
        decision_counts = feature_df["decision"].fillna("Unknown").astype(str).value_counts().reset_index()
        decision_counts.columns = ["Decision", "Count"]
        st.dataframe(decision_counts, use_container_width=True)

        show_cols = [c for c in ["feature", "family", "decision", "scope", "rationale", "evidence_note"] if c in feature_df.columns]
        keep_df = feature_df[feature_df["decision"].astype(str).str.upper().eq("KEEP")]
        rem_df = feature_df[feature_df["decision"].astype(str).str.upper().eq("REMOVE")]

        with st.expander("Features kept in the primary model", expanded=False):
            st.dataframe(keep_df[show_cols].head(250), use_container_width=True)
        with st.expander("Features removed or de-emphasized", expanded=False):
            st.dataframe(rem_df[show_cols].head(250), use_container_width=True)

        numeric_cols = [c for c in feature_df.columns if c.endswith("abs_corr_packs") or c.endswith("coef_importance") or c in ["ALL_abs_corr_packs", "site802_total_coef_importance"]]
        top_frames = []
        if "ALL_abs_corr_packs" in feature_df.columns:
            tmp = feature_df.copy()
            tmp["ALL_abs_corr_packs"] = pd.to_numeric(tmp["ALL_abs_corr_packs"], errors="coerce")
            tmp = tmp.sort_values("ALL_abs_corr_packs", ascending=False).head(25)
            top_frames.append(("Overall strongest univariate relationship to Final Alloc", tmp))
        if "site802_total_coef_importance" in feature_df.columns:
            tmp = feature_df.copy()
            tmp["site802_total_coef_importance"] = pd.to_numeric(tmp["site802_total_coef_importance"], errors="coerce")
            tmp = tmp.sort_values("site802_total_coef_importance", ascending=False).head(25)
            top_frames.append(("Site 802 specialist coefficient importance", tmp))
        for title, frame in top_frames:
            with st.expander(title, expanded=False):
                cols = [c for c in ["feature", "family", "decision", "rationale", "ALL_abs_corr_packs", "site802_total_coef_importance"] if c in frame.columns]
                st.dataframe(frame[cols], use_container_width=True)

        st.download_button(
            "Download feature decision matrix",
            feature_df.to_csv(index=False).encode("utf-8"),
            file_name="feature_decision_matrix.csv",
            mime="text/csv",
            key="download_feature_decision_matrix_deep_dive",
        )

    st.markdown("### 4. Why some features were removed")
    st.markdown(
        """
The feature pruning work removed features that were either redundant, unstable, or likely to preserve older model mistakes:

- **Old-model-output features** such as `base_pred_units`, `base_flms`, and `base_confidence` can leak old model behavior into the new model and make errors repeat.
- **Unstable DC ratios** such as `proj_to_dc` and `rec_to_dc` can become noisy when `Dc Avail` is small.
- **Duplicate demand summaries** such as max/median demand features were removed when stronger demand agreement and shortage features already captured the same idea.
- **Sparse single-signal demand flags** were de-emphasized because one isolated demand signal can create false positive allocations.

The result is a model that still has rich feature engineering, but it is less likely to overreact to noisy rows.
"""
    )

    st.markdown("### 5. How to interpret feature use in audits")
    st.markdown(
        """
When auditing an uploaded file, focus on these patterns:

- A row with **high demand agreement + low supply + positive recommendation** should usually predict a nonzero allocation.
- A row with **high `Alloc. Rec.` but low recent sales support** is a cut candidate.
- A row with **large FLM relative to demand** is risky even if one FLM is available.
- Review rows should be judged by **relative priority inside their class-line group**, not only by row-level demand.
- Site 802 and AK rows should still be audited separately, but they now use the same base Allocate and Review model paths as all other rows.
"""
    )


def show_static_feature_report():
    p = REPORTS / "SMOKE_TEST_REPORT.md"
    if p.exists():
        with st.expander("Packaged feature deep-dive text report", expanded=False):
            st.markdown(p.read_text(encoding="utf-8"))


# -----------------------------------------------------------------------------
# App boot
# -----------------------------------------------------------------------------
try:
    bundle, v35_params, v36_params, v37_params = load_bundle()
except Exception as e:
    st.error("Model bundle failed to load.")
    st.exception(e)
    st.stop()

with st.sidebar:
    st.header("Settings")
    drop_non_model = st.checkbox("Only process Allocate and Review rows", value=True)
    st.divider()
    st.caption("Model status")
    st.write("Allocate base model:", "Loaded" if bundle.get("models", {}).get("allocate") else "Missing")
    st.write("Review base model:", "Loaded" if bundle.get("models", {}).get("review") else "Missing")
    st.write("Site 802 Allocate specialist:", "Disabled - base Allocate model used")
    st.write("Site 802 Review specialist:", "Disabled - base Review model used")
    st.write("AK Allocate specialist:", "Disabled - base Allocate model used")
    st.write("AK Review specialist:", "Disabled - base Review model used")

predict_tab, audit_tab, model_tab, feature_tab, files_tab = st.tabs([
    "Predict",
    "Audit",
    "Model overview",
    "Features",
    "Files",
])

with model_tab:
    meta = bundle.get("meta", {})
    cfg = meta.get("train_config", {})
    summary = _read_json(ART / "model_summary.json")
    train_rows_display, feature_count = overview_stats(bundle)

    st.markdown("### Model summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training / audit rows", str(train_rows_display))
    c2.metric("Model input features", f"{int(feature_count):,}" if isinstance(feature_count, (int, float)) else str(feature_count))
    c3.metric("Allocate threshold", meta.get("allocate_threshold", cfg.get("allocate_threshold", "—")))
    c4.metric("Review threshold", meta.get("review_threshold", cfg.get("review_threshold", "—")))

    st.markdown("### Model paths")
    p1, p2 = st.columns(2)
    p1.info("Base Allocate model active for all Allocate rows")
    p2.info("Base Review model active for all Review rows")
    st.caption("AK and Site 802 specialist models are intentionally removed/disabled. Those rows use the base models.")

    st.markdown("### Approved worksheet inputs")
    st.write(", ".join(core.ALLOWED_FEATURES))

    with st.expander("Training configuration", expanded=False):
        st.json(cfg)
    with st.expander("Packaged model summary", expanded=False):
        st.json(summary if summary else {"status": "model_summary.json not found"})

with predict_tab:
    st.markdown("### Predict Final Alloc.")
    up = st.file_uploader("Upload allocation workbook or CSV", type=["xlsb", "xlsx", "xlsm", "xls", "csv"], key="predict_upload")
    if up:
        try:
            raw = read_upload(up)
            cleaned = drop_unnecessary_rows(raw, drop_non_model)
            canon, audit = predict_allocation(cleaned)
            output = cleaned.copy()
            target_col = find_target_column(output) or core.TARGET_COL
            output[target_col] = int_or_blank_series(audit["Predicted Final Alloc"]).values
            pred_num = numeric_units(audit["Predicted Final Alloc"])
            flags = canon["Flag"].astype(str).str.upper()
            st.success(f"Predicted {len(output):,} eligible rows.")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows", f"{len(output):,}")
            c2.metric("Nonzero allocations", f"{int((pred_num > 0).sum()):,}")
            c3.metric("Predicted units", f"{int(pred_num.sum()):,}")
            c4.metric("AK rows", f"{int(clean_site_series(canon['Site']).isin(AK_SITES).sum()):,}")

            filter_choice = st.selectbox(
                "Spot-check filter",
                [
                    "All rows",
                    "Allocate rows",
                    "Review rows",
                    "Nonzero predictions",
                    "Blank/zero predictions",
                    "Site 802 rows",
                    "AK store rows",
                ],
            )
            mask = np.ones(len(output), dtype=bool)
            masks = build_segment_masks(canon, audit)
            if filter_choice in masks:
                mask = masks[filter_choice]
            elif filter_choice == "Blank/zero predictions":
                mask = pred_num <= 0
            st.dataframe(output.loc[mask].head(1000), use_container_width=True)
            st.download_button(
                "Download filled CSV",
                output.to_csv(index=False).encode("utf-8"),
                file_name="allocation_base_allocate_review_filled_output.csv",
                mime="text/csv",
                key="download_filled_csv",
            )
            st.download_button(
                "Download audit CSV",
                audit.to_csv(index=False).encode("utf-8"),
                file_name="allocation_base_allocate_review_audit.csv",
                mime="text/csv",
                key="download_audit_csv",
            )
        except Exception as e:
            st.error("Prediction failed.")
            st.exception(e)

with audit_tab:
    st.markdown("### Audit uploaded file")
    st.markdown(
        "Upload a workbook or CSV that already has data in `Final Alloc.`. The app will predict the file, compare predictions to the existing values, and calculate smoke-test-style accuracy by model path."
    )
    audit_up = st.file_uploader("Upload file for audit", type=["xlsb", "xlsx", "xlsm", "xls", "csv"], key="audit_upload")
    if audit_up:
        try:
            raw = read_upload(audit_up)
            cleaned = drop_unnecessary_rows(raw, drop_non_model)
            target_col = find_target_column(cleaned)
            if target_col is None:
                st.warning("No `Final Alloc.` column was detected, so this file can be predicted but not audited for accuracy.")
            else:
                actual_raw = cleaned[target_col]
                has_any_actual = actual_raw.notna().any() and (actual_raw.astype(str).str.strip() != "").any()
                if not has_any_actual:
                    st.warning("The `Final Alloc.` column exists, but it appears blank. No accuracy audit can be calculated.")
                else:
                    canon, audit = predict_allocation(cleaned)
                    actual = numeric_units(actual_raw)
                    metrics = compute_audit_metrics(canon, audit, actual)
                    row_audit = build_row_audit(cleaned, canon, audit, actual)
                    all_row = metrics[metrics["Segment / model path"].eq("All rows")].iloc[0]
                    st.success("Accuracy audit completed.")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Rows audited", f"{int(all_row['Rows']):,}")
                    c2.metric("MAE units", f"{all_row['MAE Units']:.3f}")
                    c3.metric("Exact rate", f"{all_row['Exact Rate']:.2%}")
                    c4.metric("Unit delta", f"{int(all_row['Unit Delta']):,}")
                    st.markdown("### Accuracy by segment / model path")
                    display_metrics = metrics.copy()
                    for col in ["MAE Units", "RMSE Units"]:
                        display_metrics[col] = display_metrics[col].map(lambda x: f"{x:.3f}")
                    for col in ["Exact Rate", "Within 1 FLM"]:
                        display_metrics[col] = display_metrics[col].map(lambda x: f"{x:.2%}")
                    st.dataframe(display_metrics, use_container_width=True)

                    st.markdown("### Error analysis")
                    error_filter = st.selectbox(
                        "Audit row filter",
                        ["All rows", "Errors only", "False positives", "False negatives", "Site 802 rows", "AK store rows"],
                        key="audit_filter",
                    )
                    mask = np.ones(len(row_audit), dtype=bool)
                    sites = clean_site_series(canon["Site"])
                    if error_filter == "Errors only":
                        mask = row_audit["Absolute Error Units"].to_numpy() > 0
                    elif error_filter == "False positives":
                        mask = row_audit["False Positive"].to_numpy()
                    elif error_filter == "False negatives":
                        mask = row_audit["False Negative"].to_numpy()
                    elif error_filter == "Site 802 rows":
                        mask = sites.eq("802").to_numpy()
                    elif error_filter == "AK store rows":
                        mask = sites.isin(AK_SITES).to_numpy()
                    st.dataframe(row_audit.loc[mask].head(1000), use_container_width=True)
                    st.download_button(
                        "Download audit metrics CSV",
                        metrics.to_csv(index=False).encode("utf-8"),
                        file_name="uploaded_file_accuracy_metrics.csv",
                        mime="text/csv",
                        key="download_uploaded_accuracy_metrics",
                    )
                    st.download_button(
                        "Download row-level audit CSV",
                        row_audit.to_csv(index=False).encode("utf-8"),
                        file_name="uploaded_file_row_level_audit.csv",
                        mime="text/csv",
                        key="download_uploaded_row_audit",
                    )
        except Exception as e:
            st.error("Audit failed.")
            st.exception(e)

with feature_tab:
    show_feature_deep_dive(bundle)
    show_static_feature_report()

with files_tab:
    st.markdown("### Files")
    rows = []
    for p in sorted(APP_DIR.iterdir()):
        if p.is_file():
            rows.append({"file": p.name, "size_mb": round(p.stat().st_size / (1024 * 1024), 3)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
