from pathlib import Path
import pandas as pd


def _read_one(path: Path) -> pd.DataFrame:
    """Read a single spreadsheet (CSV or Excel) into a DataFrame."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def read_production_data(paths: list[Path]) -> pd.DataFrame:
    """Load and concatenate production data from multiple files."""
    frames = [_read_one(Path(p)) for p in paths]
    return pd.concat(frames, ignore_index=True)


def read_downtime_data(paths: list[Path]) -> pd.DataFrame:
    """Load and concatenate downtime data from multiple files."""
    frames = [_read_one(Path(p)) for p in paths]
    return pd.concat(frames, ignore_index=True)

