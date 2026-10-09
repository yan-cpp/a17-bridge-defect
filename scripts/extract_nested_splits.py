import shutil
import sys
import zipfile
from pathlib import Path

outer = Path(r"D:\zm\newA\a17-bridge-defect\data\raw\68827df9f367442c8be0c283e60ed3b7_V4.zip")
outdir = Path(r"D:\zm\newA\a17-bridge-defect\data\interim\gyu_det")
outdir.mkdir(parents=True, exist_ok=True)
members = ["train.zip", "valid.zip", "test.zip", "classes.txt"]
with zipfile.ZipFile(outer) as archive:
    for member in members:
        target = outdir / member
        if target.exists():
            print(f"skip {member} (exists)")
            continue
        print(f"extracting {member} ...")
        with archive.open(member) as source, open(target, "wb") as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
        print(f"done {member}")
print("ALL_DONE")
