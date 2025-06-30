import click
from pathlib import Path
import pandas as pd

from spreadsheets import read_production_data
from feature_engineering import add_recent_history
from build_time import train_build_time_model
from build_quantity import train_build_quantity_model
from defects import train_defect_model

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
    """Read raw production spreadsheets and store a parquet dataset."""
    df = read_production_data([Path(f) for f in files])
    out_dir = get_data_dir()
    df.to_parquet(out_dir / "production.parquet")
    click.echo("✅ Ingested")

@cli.command("train-build-time")
def train_build_time():
    """Train the build-time model using ingested production data."""
    data_path = get_data_dir() / "production.parquet"
    df = pd.read_parquet(data_path)
    train_build_time_model(df)
    click.echo("✅ Build-time model trained")

@cli.command("train-defects")
def train_defects():
    """Train the defect count model using ingested production data."""
    data_path = get_data_dir() / "production.parquet"
    df = pd.read_parquet(data_path)
    train_defect_model(df)
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
    """Train the build quantity model using production and downtime data."""
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
