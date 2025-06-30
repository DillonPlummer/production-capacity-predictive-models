from pathlib import Path
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from feature_engineering import add_recent_history

def train_build_time_model(df: pd.DataFrame) -> Pipeline:
    """
    Train a model to predict build_time_days given:
      - 4-week rolling avg build time
      - 4-week rolling sums of all defect columns
      - categorical columns: part_number, line, failure_mode

    df should be the raw production data (with build_start_date,
    build_complete_date, part_number, line, AND failure_mode if available).
    """
    # 1) Compute rolling features
    df_fe = add_recent_history(df)

    # 2) Dynamically find all defect roll-up columns
    roll_defs = [c for c in df_fe.columns if c.endswith("_4w_sum")]
    if not roll_defs:
        raise ValueError("No '*_4w_sum' columns found—make sure add_recent_history ran correctly.")

    # 3) Ensure 'failure_mode' exists (or create a default)
    if "failure_mode" not in df_fe.columns:
        # If the raw production data didn't have failure_mode, fill with "NONE"
        df_fe["failure_mode"] = "NONE"
    df_fe["failure_mode"] = df_fe["failure_mode"].astype(str).str.strip().str.upper()

    # 4) Assemble feature matrix and target (added "failure_mode")
    feature_cols = ["build_time_4w_avg"] + roll_defs + ["part_number", "line", "failure_mode"]
    X = df_fe[feature_cols]
    y = df_fe["build_time_days"]

    # 5) Build preprocessing + model pipeline
    preproc = ColumnTransformer(
        [
            (
                "ohe_cats",
                OneHotEncoder(handle_unknown="ignore"),
                ["part_number", "line", "failure_mode"]
            )
        ],
        remainder="passthrough"
    )
    pipe = Pipeline(
        [
            ("preproc", preproc),
            ("rf", RandomForestRegressor(n_estimators=100, random_state=42))
        ]
    )

    # 6) Train/test split and fit
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipe.fit(X_train, y_train)
    score = pipe.score(X_test, y_test)
    print(f"Build-time model R² on hold-out: {score:.3f}")

    # 7) Persist the model
    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    fname = out_dir / f"build_time_model_{timestamp}.pkl"
    joblib.dump(pipe, fname)
    print(f"Saved build-time model to {fname}")

    return pipe
