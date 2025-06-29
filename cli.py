import click
from pathlib import Path
import pandas as pd
from feature_engineering import add_recent_history
from build_time import train_build_time_model
from build_quantity import train_build_quantity_model
from defects import train_defect_model


def read_production_data(paths: list[Path]) -> pd.DataFrame:
    """Read and concatenate production spreadsheets (CSV or Excel)."""
    frames = []
    for p in paths:
        if p.suffix.lower() in {".xls", ".xlsx"}:
            frames.append(pd.read_excel(p, engine="openpyxl"))
        else:
            frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True)
    df.columns = (
        df.columns
          .str.strip()
          .str.lower()
          .str.replace(r"[ _]+", "_", regex=True)
    )
    for c in ("build_start_date", "build_complete_date"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df = df.dropna(subset=["build_start_date", "build_complete_date"])
    df["build_time_days"] = (
        df["build_complete_date"] - df["build_start_date"]
    ).dt.total_seconds() / 86400
    return df

# Paths relative to this package
PACKAGE_ROOT = Path(__file__).resolve().parent

def get_data_dir() -> Path:
    project_root = PACKAGE_ROOT.parent
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir

@click.group()
def cli():
    pass

@cli.command("ingest")
@click.argument("files", nargs=-1, type=click.Path(exists=True))
def ingest(files):
    df = read_production_data([Path(f) for f in files])
    out_dir = get_data_dir()
    df.to_parquet(out_dir / "production.parquet")
    click.echo("✅ Ingested")

@cli.command("train-build-time")
def train_build_time():
    data_path = get_data_dir() / "production.parquet"
    df = pd.read_parquet(data_path)
    df_fe = add_recent_history(df)
    train_build_time_model(df_fe)
    click.echo("✅ Build-time model trained")

@cli.command("train-defects")
def train_defects():
    data_path = get_data_dir() / "production.parquet"
    df = pd.read_parquet(data_path)
    df_fe = add_recent_history(df)
    train_defect_model(df_fe)
    click.echo("✅ Defect model trained")

@cli.command('train-build-quantity')
@click.argument('prod_files', nargs=-1, type=click.Path(exists=True))
@click.option(
    '--downtime-files', '-d',
    multiple=True,
    type=click.Path(exists=True),
    required=True,
    help="Paths to downtime sheets (xlsx or csv)."
)
def train_build_quantity(prod_files, downtime_files):
     # ingest production & downtime
    df_prod = read_production_data([Path(f) for f in prod_files])
    df_prod = add_recent_history(df_prod)
    # train and save
    train_build_quantity_model(
        df_prod,
        [Path(f) for f in downtime_files]
    )
    click.echo("✅ Build-quantity model trained")


if __name__ == "__main__":
    cli()
