from pathlib import Path
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from feature_engineering import merge_downtime_features
from spreadsheets import read_downtime_data

def train_build_quantity_model(
    df_prod: pd.DataFrame,
    downtime_paths: list[Path]
) -> Pipeline:
    """
    Train a model to predict feasible production quantity per build,
    given historical build-time features, defect rate, downtime/opportunity cost,
    plus categorical columns: part_number, line, and failure_mode.

    df_prod: production DataFrame AFTER add_recent_history(df_prod).
    downtime_paths: list of Excel/CSV files with downtime logs.
    """
    # 1) Merge in downtime features
    df_down = read_downtime_data(downtime_paths)
    df = merge_downtime_features(df_prod, df_down)

    # 2) Compute defect_rate from *all* qty_of_defect_* columns
    defect_cols = [
        c for c in df.columns
        if c.startswith("qty_of_defect_") and not c.endswith("_4w_sum")
    ]
    if not defect_cols:
        raise ValueError("No defect columns found in DataFrame!")
    df["total_defects"] = df[defect_cols].sum(axis=1)
    df["defect_rate"]    = df["total_defects"] / df["qty_produced"]

    # 3) Select features & target (added "line" and "failure_mode")
    features = [
        "build_time_days",     # actual last build time
        "build_time_4w_avg",   # 4-week rolling avg
        "defect_rate",         # historical defect rate
        "downtime_min",        # downtime during this build
        "part_number",         # one-hot encoded
        "line",                # one-hot encoded
        "failure_mode"         # one-hot encoded
    ]
    X = df[features]
    y = df["qty_produced"]

    # 4) Build & train the pipeline
    preproc = ColumnTransformer(
        [(
            "ohe_cats",
            OneHotEncoder(handle_unknown="ignore"),
            ["part_number", "line", "failure_mode"]
        )],
        remainder="passthrough"
    )
    pipe = Pipeline([
        ("preproc", preproc),
        ("rf", RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipe.fit(X_train, y_train)

    # 5) Report & persist
    score = pipe.score(X_test, y_test)
    print(f"Build Quantity R² on hold-out: {score:.3f}")

    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)
    fname = out_dir / f"build_quantity_model_{pd.Timestamp.now():%Y%m%d_%H%M}.pkl"
    joblib.dump(pipe, fname)
    print(f"Saved model to {fname}")

    return pipe
