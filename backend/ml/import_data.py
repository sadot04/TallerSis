import argparse
from pathlib import Path
import shutil
from zipfile import ZipFile

cli = argparse.ArgumentParser(description="Extrae train.csv y test.csv del ZIP de la competencia.")
cli.add_argument("zip", type=Path)
args = cli.parse_args()
target = Path(__file__).resolve().parents[1] / "data"
target.mkdir(exist_ok=True)
with ZipFile(args.zip) as archive:
    for name in ["train.csv", "test.csv"]:
        with archive.open(name) as source, (target / name).open("wb") as output:
            shutil.copyfileobj(source, output)
print(target)
