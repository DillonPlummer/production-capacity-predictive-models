import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from lime.lime_tabular import LimeTabularExplainer
from feature_engineering import add_recent_history, merge_downtime_features
from upsetplot import from_indicators, UpSet



# Page config & figure sizing
st.set_page_config(layout="centered")
sns.set(rc={"figure.figsize": (16, 9)})

# Ensure project root on path
project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))

def get_latest_model(pattern: str) -> Path:
    files = list((project_root / "models").glob(pattern))
    if not files:
        st.error(f"No models found for pattern `{pattern}`")
        st.stop()
    return max(files, key=lambda p: p.stat().st_mtime)

st.title("QualityLab ML Dashboard (Seaborn)")

# — Sidebar upload form w/ session state —
if "uploaded" not in st.session_state:
    st.session_state.uploaded = False

with st.sidebar.form("upload_form"):
    st.header("Upload Data")
    prod_files = st.file_uploader(
        "Production sheets", type=["xlsx", "xls", "csv"],
        accept_multiple_files=True, key="prod"
    )
    down_files = st.file_uploader(
        "Downtime sheets", type=["xlsx", "xls", "csv"],
        accept_multiple_files=True, key="down"
    )
    plan_file = st.file_uploader(
        "Build Plan", type=["xlsx", "xls", "csv"],
        accept_multiple_files=False, key="plan"
    )
    submitted = st.form_submit_button("Submit")
    if submitted:
        if not (st.session_state.prod and st.session_state.down and st.session_state.plan):
            st.warning("Please upload production, downtime AND plan files.")
        else:
            st.session_state.uploaded = True

if not st.session_state.uploaded:
    st.sidebar.info("Upload all three files and click Submit.")
    st.stop()

# — Ingest & clean production data —
raw_prod = []
for up in st.session_state.prod:
    if up.name.lower().endswith(("xlsx", "xls")):
        raw_prod.append(pd.read_excel(up, engine="openpyxl"))
    else:
        raw_prod.append(pd.read_csv(up))
df_prod = pd.concat(raw_prod, ignore_index=True)
df_prod.columns = (
    df_prod.columns
           .str.strip()
           .str.lower()
           .str.replace(r"[ _]+", "_", regex=True)
)
# parse key dates
for c in ("build_start_date", "build_complete_date"):
    df_prod[c] = pd.to_datetime(df_prod[c], errors="coerce")
df_prod = df_prod.dropna(subset=["build_start_date", "build_complete_date"])
df_prod["build_time_days"] = (
    df_prod["build_complete_date"] - df_prod["build_start_date"]
).dt.total_seconds() / 86400

# enforce & normalize line
if "line" not in df_prod.columns:
    st.error("Missing 'line' in production data")
    st.stop()
df_prod["line"] = df_prod["line"].astype(str).str.strip().str.upper()
valid_lines = df_prod["line"].unique().tolist()

# enforce & normalize part_number
if "part_number" not in df_prod.columns:
    st.error("Missing 'part_number' in production data")
    st.stop()
df_prod["part_number"] = df_prod["part_number"].astype(str).str.strip()

# enforce & normalize any qty_of_defect_* columns remain numeric
orig_defs = [c for c in df_prod.columns if c.startswith("qty_of_defect_")]
if not orig_defs:
    st.error("No defect columns found in production data")
    st.stop()
for c in orig_defs:
    df_prod[c] = pd.to_numeric(df_prod[c], errors="coerce").fillna(0)

# — Feature engineering on prod history —
df_fe = add_recent_history(df_prod)

# — Ingest & clean downtime data —
raw_down = []
for up in st.session_state.down:
    if up.name.lower().endswith(("xlsx", "xls")):
        raw_down.append(pd.read_excel(up, engine="openpyxl"))
    else:
        raw_down.append(pd.read_csv(up))
df_down = pd.concat(raw_down, ignore_index=True)
df_down.columns = (
    df_down.columns
           .str.strip()
           .str.lower()
           .str.replace(r"[ _]+", "_", regex=True)
)
if "date" not in df_down.columns:
    st.error("Missing 'date' in downtime data")
    st.stop()
df_down["date"] = pd.to_datetime(df_down["date"], errors="coerce")
df_down = df_down.dropna(subset=["date"])
if "line" not in df_down.columns:
    st.error("Missing 'line' in downtime data")
    st.stop()
df_down["line"] = df_down["line"].astype(str).str.strip().str.upper()
mask_bad = ~df_down["line"].isin(valid_lines)
if mask_bad.any():
    st.warning(f"Dropping {mask_bad.sum()} downtime rows on lines not in production")
    df_down = df_down.loc[~mask_bad].copy()

# Normalize failure_mode for downtime
if "failure_mode" in df_down.columns:
    df_down["failure_mode"] = (
        df_down["failure_mode"]
          .astype(str)
          .str.strip()
          .str.upper()
          .str.replace(r"^\d+\s*-\s*", "", regex=True)
    )
else:
    df_down["failure_mode"] = "NONE"

# — Merge downtime into history features —
df_fe = merge_downtime_features(df_fe, df_down)

# ### DEBUG TEST ###
# # (after merging)
# counts = df_fe["failure_mode"].value_counts().reset_index()
# counts.columns = ["failure_mode", "count"]
# st.write("How many builds got each failure_mode after merging:")
# st.dataframe(counts)
# ###

# — Save a copy of full, unfiltered history —
df_fe_full = df_fe.copy()

# — Dynamically discover defect columns & defect-4w sums —
orig_defs = [
    c for c in df_fe.columns
    if c.startswith("qty_of_defect_") and not c.endswith("_4w_sum")
]
defect_4w = [f"{c}_4w_sum" for c in orig_defs]

# recompute defect_rate for quantity model
df_fe["total_defects"] = df_fe[orig_defs].sum(axis=1)
df_fe["defect_rate"]    = df_fe["total_defects"] / df_fe["qty_produced"]

# — Ingest & clean build plan —
up = st.session_state.plan
if up.name.lower().endswith(("xlsx", "xls")):
    df_plan = pd.read_excel(up, engine="openpyxl")
else:
    df_plan = pd.read_csv(up)
df_plan.columns = (
    df_plan.columns
           .str.strip()
           .str.lower()
           .str.replace(r"[ _]+", "_", regex=True)
)
df_plan["plan_start_date"] = pd.to_datetime(df_plan["plan_start_date"], errors="coerce")
if "plan_end_date" in df_plan.columns:
    df_plan["plan_end_date"] = pd.to_datetime(df_plan["plan_end_date"], errors="coerce")
else:
    df_plan["plan_end_date"] = df_plan["plan_start_date"]
auto_bt = (df_plan["plan_end_date"] - df_plan["plan_start_date"]).dt.total_seconds() / 86400
df_plan["build_time_days"] = df_plan.get("build_time_days", auto_bt)
df_plan = df_plan.dropna(subset=["part_number", "planned_qty", "plan_start_date", "build_time_days"])
if "line" not in df_plan.columns:
    st.error("Missing 'line' in build plan")
    st.stop()
df_plan["line"] = df_plan["line"].astype(str).str.strip().str.upper()
df_plan["part_number"] = df_plan["part_number"].astype(str).str.strip()

### DEBUG TEST ###
st.write("🗓️ Production date range:",
         df_prod["build_start_date"].min(), "→", df_prod["build_start_date"].max())
st.write("🗓️ Plan start dates:",
         df_plan["plan_start_date"].min(), "→", df_plan["plan_start_date"].max())

# — Build plan features dynamically, include real 4-week defect sums & failure_mode —
def make_plan_features(r):
    pn    = r["part_number"]
    st_dt = r["plan_start_date"]
    ln    = r["line"]

    # 1) Historical slice for this part in last 28 days
    hist = df_prod[
        (df_prod["part_number"] == pn)
        & (df_prod["build_start_date"] >= st_dt - pd.Timedelta(days=28))
        & (df_prod["build_start_date"] < st_dt)
    ]

    # Sum each defect column over that slice
    defect_4w_vals = {f"{c}_4w_sum": hist[c].sum() for c in orig_defs}

    # Compute 4-week average build time
    bt_4w = hist["build_time_days"].mean() if not hist.empty else 0.0

    # 2) Downtime and opportunity cost during this build plan
    down_mask = (
        (df_down["line"] == ln)
        & (df_down["date"] >= st_dt)
        & (df_down["date"] <= r["plan_end_date"])
    )

    segment = df_down[
        (df_down.line == ln) &
        (df_down.date.between(st_dt, r.plan_end_date))
    ]

    dt_sum = segment.downtime_min.sum()
    # ALL unique modes
    modes = segment.failure_mode.dropna().unique().tolist()
    mode_list = ", ".join(modes) if modes else "NONE"

    # 3) Determine failure_mode exactly as merge_downtime_features does:
    sub = df_down[df_down["line"] == ln]
    fm_mask = (sub["date"] >= st_dt) & (sub["date"] <= r["plan_end_date"])
    matched = sub.loc[fm_mask, "failure_mode"]

    if not matched.empty:
        most_common = matched.mode()
        if not most_common.empty:
            failure_for_row = most_common.iloc[0]
        else:
            failure_for_row = matched.iloc[0]
    else:
        failure_for_row = "NONE"

    failure_for_row = str(failure_for_row).strip().upper()

    # 4) Build the feature dict, **including** plan_start_date (and optionally plan_end_date)
    feat = {
        "part_number":       pn,
        "line":              ln,
        "planned_qty":       r["planned_qty"],
        "plan_start_date":   st_dt,
        "plan_end_date":     r["plan_end_date"],
        "build_time_days":   r["build_time_days"],
        "build_time_4w_avg": bt_4w,
        "defect_rate":       hist[orig_defs].sum(axis=1).sum() / max(1, len(hist)),
        "downtime_min":      dt_sum,
        "failure_mode":      segment.failure_mode.mode().iloc[0] if not segment.empty else "NONE",
        "failure_modes":     mode_list,   # <— plural, comma-joined
        **defect_4w_vals
    }
    # Inject each qty_of_defect_X_4w_sum
    feat.update(defect_4w_vals)

    return feat

# If there's plan data, build df_plan_feats; otherwise create an empty DataFrame with the correct columns:
if not df_plan.empty:
    df_plan_feats = pd.DataFrame([make_plan_features(r) for _, r in df_plan.iterrows()])
    ### DEBUG TEST ###
    st.write("Plan-feature debug")
    st.dataframe(
        df_plan_feats[
        ['part_number','plan_start_date','build_time_4w_avg','downtime_min']
        ],
        use_container_width=True
    )
else:
    # When plan is empty, define every column that make_plan_features would have returned:
    df_plan_feats = pd.DataFrame(columns=[
    "part_number","line","planned_qty","plan_start_date","plan_end_date",
    "build_time_days","build_time_4w_avg","defect_rate","downtime_min",
    "failure_mode","failure_modes",
    *defect_4w
])
if "failure_mode" not in df_plan_feats.columns:
    df_plan_feats["failure_mode"] = "NONE"
else:
    df_plan_feats["failure_mode"] = df_plan_feats["failure_mode"].astype(str).str.strip().str.upper()

# ─────────────────────────────────────────────────────────────────────────────

# — Load models & extract expected feature names —
build_model   = joblib.load(get_latest_model("build_time_model_*.pkl"))
defect_model  = joblib.load(get_latest_model("defect_model_*.pkl"))
qty_model     = joblib.load(get_latest_model("build_quantity_model_*.pkl"))

# ─── Extract feature‐lists exactly as used by each model ─────────────────────
bt_feats         = list(build_model.feature_names_in_)    # e.g. [..., "part_number", "line", "failure_mode"]
tq_feats         = list(qty_model.feature_names_in_)      # e.g. [..., "part_number", "line", "failure_mode"]
def_model_feats  = list(defect_model.feature_names_in_)   # needed for defect predictions

# ─── (Re-)predict on df_fe so df_fe[tq_feats], df_fe[bt_feats], and pred_{defects} exist ─────────
# Build-time predictions on history
for c in bt_feats:
    if c not in df_fe.columns:
        df_fe[c] = 0.0
df_fe["pred_build_time"] = build_model.predict(df_fe[bt_feats])

# Quantity predictions on history
for c in tq_feats:
    if c not in df_fe.columns:
        df_fe[c] = 0.0
df_fe["pred_quantity"] = qty_model.predict(df_fe[tq_feats])

# Defect-count predictions (multi-output) on history
for c in def_model_feats:
    if c not in df_fe.columns:
        df_fe[c] = 0.0
preds = defect_model.predict(df_fe[def_model_feats])
for i, col in enumerate(orig_defs):
    df_fe[f"pred_{col}"] = preds[:, i]

# ─── Predict on df_plan_feats (if not empty) ───────────────────────────────────
if not df_plan_feats.empty:
    # 1) Split bt_feats into numeric vs categorical
    num_bt_feats = [c for c in bt_feats if c not in ("part_number", "line", "failure_mode")]
    cat_bt_feats = ["part_number", "line", "failure_mode"]

    # 2) Coerce numeric to floats
    for c in num_bt_feats:
        if c not in df_plan_feats.columns:
            df_plan_feats[c] = 0.0
        else:
            df_plan_feats[c] = pd.to_numeric(df_plan_feats[c], errors="coerce").fillna(0.0)

    # 3) Ensure categorical remain strings
    for c in cat_bt_feats:
        if c not in df_plan_feats.columns:
            df_plan_feats[c] = "NONE"
        else:
            df_plan_feats[c] = df_plan_feats[c].astype(str).str.strip().str.upper()

    # 4) Now we can safely call build_model.predict(...)
    df_plan_feats["pred_build_time"] = build_model.predict(df_plan_feats[bt_feats])

    # — Repeat for quantity model —
    num_tq_feats = [c for c in tq_feats if c not in ("part_number", "line", "failure_mode")]
    cat_tq_feats = ["part_number", "line", "failure_mode"]

    for c in num_tq_feats:
        if c not in df_plan_feats.columns:
            df_plan_feats[c] = 0.0
        else:
            df_plan_feats[c] = pd.to_numeric(df_plan_feats[c], errors="coerce").fillna(0.0)

    for c in cat_tq_feats:
        if c not in df_plan_feats.columns:
            df_plan_feats[c] = "NONE"
        else:
            df_plan_feats[c] = df_plan_feats[c].astype(str).str.strip().str.upper()

    df_plan_feats["pred_qty"] = qty_model.predict(df_plan_feats[tq_feats])

else:
    df_plan_feats["pred_build_time"] = pd.Series(dtype=float)
    df_plan_feats["pred_qty"]        = pd.Series(dtype=float)

# ─── Encode categorical columns as integer codes for LIME ─────────────────────
# part_number
df_fe["part_number_code"] = df_fe["part_number"].astype("category").cat.codes
part_number_categories = dict(enumerate(
    df_fe["part_number"].astype("category").cat.categories
))
inv_part_number = {v: k for k, v in part_number_categories.items()}

# line
df_fe["line_code"] = df_fe["line"].astype("category").cat.codes
line_categories = dict(enumerate(
    df_fe["line"].astype("category").cat.categories
))
inv_line = {v: k for k, v in line_categories.items()}

# failure_mode
df_fe["failure_mode_code"] = df_fe["failure_mode"].astype("category").cat.codes
failure_mode_categories = dict(enumerate(
    df_fe["failure_mode"].astype("category").cat.categories
))
inv_failure_mode = {v: k for k, v in failure_mode_categories.items()}

# ─── Build LIME feature lists ──────────────────────────────────────────────────
numeric_bt = [c for c in bt_feats if c not in ("part_number", "line", "failure_mode")]
numeric_tq = [c for c in tq_feats if c not in ("part_number", "line", "failure_mode")]

lime_bt_feats = numeric_bt + ["part_number_code", "line_code", "failure_mode_code"]
lime_tq_feats = numeric_tq + ["part_number_code", "line_code", "failure_mode_code"]

# Categorical column indices (for LIME)
cat_indices_bt = [
    lime_bt_feats.index("part_number_code"),
    lime_bt_feats.index("line_code"),
    lime_bt_feats.index("failure_mode_code"),
]
cat_indices_tq = [
    lime_tq_feats.index("part_number_code"),
    lime_tq_feats.index("line_code"),
    lime_tq_feats.index("failure_mode_code"),
]

# Categorical names mapping index → list of categories
categorical_names_bt = {
    lime_bt_feats.index("part_number_code"): list(part_number_categories.values()),
    lime_bt_feats.index("line_code"): list(line_categories.values()),
    lime_bt_feats.index("failure_mode_code"): list(failure_mode_categories.values()),
}
categorical_names_tq = {
    lime_tq_feats.index("part_number_code"): list(part_number_categories.values()),
    lime_tq_feats.index("line_code"): list(line_categories.values()),
    lime_tq_feats.index("failure_mode_code"): list(failure_mode_categories.values()),
}

# ─── Create LIME explainers with categorical awareness ─────────────────────────
qty_explainer = LimeTabularExplainer(
    training_data=df_fe[lime_tq_feats].values,
    feature_names=lime_tq_feats,
    categorical_features=cat_indices_tq,
    categorical_names=categorical_names_tq,
    mode="regression"
)

time_explainer = LimeTabularExplainer(
    training_data=df_fe[lime_bt_feats].values,
    feature_names=lime_bt_feats,
    categorical_features=cat_indices_bt,
    categorical_names=categorical_names_bt,
    mode="regression"
)

# ─── Define feature_name_map & pretty_feat ────────────────────────────────────
feature_name_map = {
    "build_time_days":    "build time (days)",
    "build_time_4w_avg":  "historical build time",
    "defect_rate":        "defect rate",
    "downtime_min":       "downtime",
    # You can extend this map for any additional numeric features if desired
}

def pretty_feat(feat):
    return feature_name_map.get(feat, feat)

# ─── Sidebar: Filter Form ─────────────────────────────────────────────────────
all_parts = sorted(df_fe["part_number"].unique())

# 1) Initialize session‐state defaults (before any widgets):
if "filters_applied" not in st.session_state:
    st.session_state.filters_applied = False
if "filter_mode_radio" not in st.session_state:
    st.session_state.filter_mode_radio = "Custom"
if "selected_parts" not in st.session_state:
    # on very first load, select everything
    st.session_state.selected_parts = all_parts.copy()

# 2) Sidebar form
with st.sidebar.form("filter_form"):
    st.header("Filter Data")

    # Date range
    dates = df_prod["build_start_date"].dt.date
    min_d, max_d = dates.min(), dates.max()
    start_d, end_d = st.date_input(
        "Build start date range",
        value=(min_d, max_d),
        min_value=min_d
    )

    # Selection mode radio
    mode = st.radio(
        "Selection mode",
        options=["Custom", "Select All"],
        key="filter_mode_radio"
    )

    # Multiselect: default depends on mode
    default = all_parts if mode == "Select All" else st.session_state.selected_parts
    chosen = st.multiselect(
        "Choose part number(s)",
        options=all_parts,
        default=default
    )

    apply = st.form_submit_button("Apply Filters")

# 3) When they click Apply, write back to the same key
if apply:
    if st.session_state.filter_mode_radio == "Select All":
        st.session_state.selected_parts = all_parts.copy()
    else:  # Custom
        st.session_state.selected_parts = chosen
    st.session_state.filters_applied = True

# 4) If filters not yet applied, block until they do
if not st.session_state.filters_applied:
    st.sidebar.info("Set your filters above and click **Apply Filters**.")
    st.stop()

# 5) Finally apply the filter to df_fe
df_fe = df_fe.loc[
    (df_fe["build_start_date"].dt.date >= start_d)
    & (df_fe["build_start_date"].dt.date <= end_d)
    & (df_fe["part_number"].isin(st.session_state.selected_parts))
]





# ─── Tabs ──────────────────────────────────────────────────────────────────
tabs = st.tabs([
        "Model Performance",
        "Capacity Predictions",
        "Descriptive Statistics",
        "Co-occurrences"
    ])
t1, t2, t3, t4 = tabs

# ─── Tab 1: Model Performance ──────────────────────────────────────────────────────
with t1:
    fig, ax = plt.subplots()
    sns.scatterplot(
        data=df_fe,
        x="build_time_days",
        y="pred_build_time",
        hue="part_number" if len(st.session_state.selected_parts) > 1 else None,
        ax=ax
    )
    ax.plot(
        [df_fe["build_time_days"].min(), df_fe["build_time_days"].max()],
        [df_fe["build_time_days"].min(), df_fe["build_time_days"].max()],
        "--", color="gray"
    )
    ax.set(xlabel="Actual Build Time (days)", ylabel="Predicted Build Time (days)")
    st.pyplot(fig, use_container_width=False)
    for col in orig_defs:
        actual = col
        pred   = f"pred_{col}"
        dfm = df_fe.melt(
            id_vars=["build_start_date"],
            value_vars=[actual, pred],
            var_name="type", value_name="count"
        )
        fig, ax = plt.subplots()
        sns.lineplot(data=dfm, x="build_start_date", y="count", hue="type", ax=ax)
        ax.set_ylabel(col.replace("qty_of_defect_", "Defect ").title())
        st.pyplot(fig, use_container_width=False)


# ─── Tab 2: Capacity Predictions ─────────
with t2:
    st.subheader("Capacity Predictions")

    if df_plan_feats.empty:
        st.info("No plan data available for your chosen part(s).")
    else:
        # loop over each plan‐row
        for _, r in df_plan_feats.iterrows():
            pn          = r["part_number"]
            planned_qty = r["planned_qty"]
            pred_qty    = r["pred_qty"]
            planned_bt  = r["build_time_days"]
            pred_bt     = r["pred_build_time"]
            st_dt       = r["plan_start_date"]

            st.write(f"**Part {pn}**")
            delta_qty   = planned_qty - pred_qty
            st.write(f"- Planned Qty: {planned_qty:.0f}")
            st.write(f"- Predicted capacity: {pred_qty:.0f} parts")
            if delta_qty > 0:
                st.write(f"- Shortfall of {delta_qty:.0f} parts.")

                # build LIME vector
                pn_code   = inv_part_number[r["part_number"]]
                line_code = inv_line[r["line"]]
                fm_code   = inv_failure_mode[r["failure_mode"]]
                vec       = [r[c] for c in numeric_tq] + [pn_code, line_code, fm_code]

                exp_qty = qty_explainer.explain_instance(
                    np.array(vec),
                    predict_fn=lambda Xn: qty_model.predict(
                        pd.DataFrame(Xn, columns=lime_tq_feats)
                          .assign(part_number=lambda df: df["part_number_code"].map(part_number_categories))
                          .assign(line=lambda df: df["line_code"].map(line_categories))
                          .assign(failure_mode=lambda df: df["failure_mode_code"].map(failure_mode_categories))
                          .drop(columns=["part_number_code","line_code","failure_mode_code"])
                    ),
                    num_features=15
                )

                # get only the negative‐weight features, *excluding* part_number_code & NONE failure_mode
                neg = [
                    (feat, -w)
                    for feat, w in exp_qty.as_list()
                    if w < 0
                    and not feat.startswith("part_number_code")
                    and not (feat.startswith("failure_mode_code") and " = NONE" in feat)
                ]
                total = sum(w for _, w in neg) or 1e-9

                rows = []
                for feat_name, raw_loss in neg:
                    scaled = raw_loss/total * delta_qty
                    pct    = scaled/delta_qty*100

                    # unwrap categorical encoding
                    if " = " in feat_name:
                        code, cat = feat_name.split(" = ")
                        raw_col = code.replace("_code","")
                    else:
                        raw_col, cat = feat_name, None

                    # for downtime, pull the actual modes you stored
                    if raw_col == "downtime_min":
                        # r["failure_modes"] is the comma-joined list you built above
                        reason = f"{pretty_feat(raw_col)} (mode(s): {r['failure_modes']})"
                    elif cat is not None:
                        reason = f"{pretty_feat(raw_col)} = {cat}"
                    else:
                        reason = pretty_feat(raw_col)


                    rows.append({
                        "Reason":        reason,
                        "Parts Lost":    int(round(scaled)),
                        "% of Shortfall": f"{pct:.1f}%"
                    })

                st.write("**Quantity Shortfall Breakdown**")
                st.table(pd.DataFrame(rows).sort_values("Parts Lost", ascending=False))

            # … your build‐time overrun section follows similarly …
            st.write("---")


# ─── Tab 3: Descriptive Statistics ───────────────────────────────────────────
with t3:
    st.subheader("Descriptive Statistics")

    # 1) Summary statistics (mean, median, mode, std, min, max)
    numeric_cols = df_fe.select_dtypes(include="number").columns.tolist()
    desc = df_fe[numeric_cols].describe().T
    desc = desc.rename(columns={"50%": "median"})
    mode = df_fe[numeric_cols].mode().iloc[0]
    desc["mode"] = mode
    st.write("**Summary Statistics**")
    st.dataframe(desc, use_container_width=True)

    # 2) Bar chart: avg qty_produced per part_number
    st.write("**Average Quantity Produced per Part**")
    avg_qty = (
        df_fe
        .groupby("part_number")["qty_produced"]
        .mean()
        .reset_index()
        .sort_values("qty_produced", ascending=False)
    )
    fig, ax = plt.subplots()
    sns.barplot(data=avg_qty, x="part_number", y="qty_produced", ax=ax)
    ax.set_xlabel("Part Number")
    ax.set_ylabel("Avg Qty Produced")
    st.pyplot(fig, use_container_width=True)

    # 3) Failure-Mode Counts per Part ─────────────────────────
    st.write("**Failure Mode Counts per Part**")
    # consider only actual downtime events
    df_fail = df_fe[df_fe['failure_mode'] != 'NONE']
    df_fail_counts = (
        df_fail
        .groupby(['part_number', 'failure_mode'])
        .size()
        .reset_index(name='count')
    )
    fig, ax = plt.subplots()
    sns.barplot(
        data=df_fail_counts,
        x='part_number', y='count', hue='failure_mode',
        dodge=True, ax=ax
    )
    ax.set_xlabel("Part Number")
    ax.set_ylabel("Number of Downtime Events")
    ax.legend(title="Failure Mode", bbox_to_anchor=(1,1))
    st.pyplot(fig, use_container_width=True)

    # 4) Line chart: weekly average defect_rate over time
    st.write("**Weekly Avg Defect Rate Over Time**")
    ts = (
        df_fe
        .set_index("build_start_date")["defect_rate"]
        .resample("W")
        .mean()
        .reset_index()
    )
    fig, ax = plt.subplots()
    sns.lineplot(data=ts, x="build_start_date", y="defect_rate", ax=ax)
    ax.set_xlabel("Week Ending")
    ax.set_ylabel("Avg Defect Rate")
    st.pyplot(fig, use_container_width=True)

    # 5) Pareto chart: total defects by type + cumulative %
    st.write("**Pareto Chart of Defect Counts**")
    defect_totals = df_fe[orig_defs].sum().sort_values(ascending=False)
    cumperc = defect_totals.cumsum() / defect_totals.sum() * 100

    fig, ax1 = plt.subplots()
    defect_labels = [c.replace("qty_of_defect_", "") for c in defect_totals.index]
    ax1.bar(defect_labels, defect_totals.values)
    ax1.set_xlabel("Defect Type")
    ax1.set_ylabel("Total Count")
    ax1.tick_params(axis="x", rotation=45)

    ax2 = ax1.twinx()
    ax2.plot(defect_labels, cumperc.values, marker="o")
    ax2.set_ylabel("Cumulative %")
    st.pyplot(fig, use_container_width=True)

    # 6) Defect co-occurrence heatmap (only raw defect counts)
    st.write("**Defect Co-occurrence Heatmap**")
    raw_defs = orig_defs
    corr     = df_fe[raw_defs].corr()

    # create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots()
    sns.heatmap(
        corr,
        mask=mask,               # <-- only show lower triangle
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        linewidths=0.5,
        ax=ax
    )
    labels = [c.replace("qty_of_defect_", "") for c in raw_defs]
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels, rotation=0)
    st.pyplot(fig, use_container_width=True)

# ─── Tab 4: Co-occurrence Visualizations ─────────────────────────────────────
with t4:
    st.subheader("Co-occurrence Visualizations")

    # 1) 100%-Stacked Bar
    st.write("**100%-Stacked Bar**")
    base_defs = orig_defs
    ncols = 2
    nrows = (len(base_defs) + ncols - 1) // ncols

    # Enable constrained_layout so matplotlib adds padding automatically
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(12, 4 * nrows),
        constrained_layout=True
    )
    axes = axes.flatten()

    for ax_i, fd in zip(axes, base_defs):
        mask_fd = df_fe[fd] > 0
        co_props = (
            df_fe.loc[mask_fd, base_defs]
                 .gt(0)
                 .mean()
                 .sort_values(ascending=False)
        )
        labels = [c.replace("qty_of_defect_", "") for c in co_props.index]

        # Set bar height smaller so they don’t touch
        ax_i.barh(labels, co_props.values, height=0.6)

        ax_i.set_xlim(0, 1)
        ax_i.set_title(fd.replace("qty_of_defect_", ""))
        ax_i.set_xlabel("Proportion of builds with this defect")

    # turn off any unused axes
    for ax_unused in axes[len(base_defs):]:
        ax_unused.axis("off")

    st.pyplot(fig, use_container_width=True)

    # 2) Weekly Stacked-Area
    st.write("**Weekly Stacked-Area**")
    focus = base_defs[0]
    weekly = (
        df_fe.assign(week=lambda d: d.build_start_date.dt.to_period("W").dt.start_time)
             .groupby("week")
             .apply(lambda g: g[g[focus] > 0][orig_defs].gt(0).mean())
             .reset_index()
             .melt(id_vars="week", var_name="defect", value_name="prop")
    )
    fig, ax = plt.subplots()
    for dname, grp in weekly.groupby("defect"):
        ax.fill_between(grp["week"], grp["prop"], label=dname.replace("qty_of_defect_", ""))
    ax.legend(title="Co-Defect", bbox_to_anchor=(1, 1))
    ax.set(xlabel="Week", ylabel="Proportion")
    st.pyplot(fig, use_container_width=True)

    # # 3) UpSet Plot
    # st.write("**UpSet Plot**")
    # bools = df_fe[orig_defs].gt(0).rename(columns=lambda c: c.replace("qty_of_defect_", ""))
    # combo_counts = from_indicators(bools)
    # fig = plt.figure(figsize=(8, 6))
    # UpSet(combo_counts, subset_size="count", show_counts=True).plot(fig=fig)
    # st.pyplot(fig, use_container_width=True)

# — Raw data preview —
st.subheader("Production Data & Predictions")
st.dataframe(df_fe, use_container_width=False)
