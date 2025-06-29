import pandas as pd
import joblib
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split

from qualitylab.ml.feature_engineering import add_recent_history

def train_defect_model(df: pd.DataFrame) -> Pipeline:
    """
    Train a multi-output model to predict counts for each defect type.
    - df: raw production DataFrame (must include qty_of_defect_* columns,
          build_start_date, build_complete_date, part_number, line,
         etc.)
    """
    # 1) Add 4-week rolling history
    df_fe = add_recent_history(df)

    # 2) Discover your defect columns dynamically
    orig_defs = [
        c for c in df_fe.columns
        if c.startswith("qty_of_defect_") and not c.endswith("_4w_sum")
    ]
    if not orig_defs:
        raise ValueError("No 'qty_of_defect_*' columns found in DataFrame!")
    roll_defs = [f"{c}_4w_sum" for c in orig_defs]

    # 3) Build feature & target matrices (added "line" and "part_number")
    feature_cols = ["build_time_days", "build_time_4w_avg"] + roll_defs + ["part_number", "line"]
    X = df_fe[feature_cols]
    y = df_fe[orig_defs]  # DataFrame with one column per defect

    # 4) Preprocessing + model pipeline
    preproc = ColumnTransformer(
        [
            (
                "ohe_cats",
                OneHotEncoder(handle_unknown="ignore"),
                ["part_number", "line"]
            )
        ],
        remainder="passthrough"
    )
    pipe = Pipeline(
        [
            ("preproc", preproc),
            (
                "multi_rf",
                MultiOutputRegressor(
                    RandomForestRegressor(n_estimators=100, random_state=42)
                )
            ),
        ]
    )

    # 5) Train/test split & fit
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipe.fit(X_train, y_train)

    # 6) Report performance
    r2 = pipe.score(X_test, y_test)
    print(f"Defect model R² scores (averaged across outputs): {r2:.3f}")

    # 7) Persist the model artifact
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    out_path  = Path("models") / f"defect_model_{timestamp}.pkl"
    out_path.parent.mkdir(exist_ok=True)
    joblib.dump(pipe, out_path)
    print(f"Saved defect model to {out_path}")

    return pipe
