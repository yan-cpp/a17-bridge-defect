import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

URL = "https://dacl10k.s3.eu-central-1.amazonaws.com/dacl10k-challenge/dacl10k_v2_devphase.zip"
DEST = Path(r"D:\zm\newA\a17-bridge-defect\data\raw\dacl10k_v2_devphase.zip")
CONNECTIONS = 16

session = requests.Session()
head = session.head(URL, timeout=60, allow_redirects=True)
total = int(head.headers.get("Content-Length", "0"))
if total <= 0:
    raise SystemExit(f"Content-Length unavailable: {total}")

chunk_size = (total + CONNECTIONS - 1) // CONNECTIONS
tmp_dir = DEST.parent / ".dacl_parts"
tmp_dir.mkdir(parents=True, exist_ok=True)
print(f"downloading {total} bytes in {CONNECTIONS} parts", flush=True)


def fetch(index: int, start: int, end: int) -> str:
    part = tmp_dir / f"part_{index:02d}"
    if part.exists() and part.stat().st_size == (end - start + 1):
        return str(part)
    for attempt in range(3):
        try:
            headers = {"Range": f"bytes={start}-{end}"}
            with session.get(URL, headers=headers, timeout=120, stream=True) as response:
                response.raise_for_status()
                with open(part, "wb") as handle:
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        if block:
                            handle.write(block)
            if part.stat().st_size == (end - start + 1):
                return str(part)
        except Exception as exc:
            print(f"part {index} attempt {attempt + 1} failed: {exc}", flush=True)
            if part.exists():
                part.unlink()
    raise RuntimeError(f"part {index} failed")

ranges = []
for i in range(CONNECTIONS):
    start = i * chunk_size
    end = min(start + chunk_size - 1, total - 1)
    if start >= total:
        break
    ranges.append((i, start, end))

parts = []
with ThreadPoolExecutor(max_workers=CONNECTIONS) as pool:
    futures = [pool.submit(fetch, i, start, end) for i, start, end in ranges]
    for future in as_completed(futures):
        parts.append(future.result())

parts.sort(key=lambda name: int(Path(name).stem.split("_")[1]))
with open(DEST, "wb") as output:
    for name in parts:
        with open(name, "rb") as handle:
            output.write(handle.read())
        Path(name).unlink()

print("download complete", DEST, DEST.stat().st_size, flush=True)
