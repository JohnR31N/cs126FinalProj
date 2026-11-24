"""Cache the official CIFAR-10 archive over short, independently checked range requests."""
import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import NamedTemporaryFile

import requests

from external_data import BASE, FILES
from idec_final.utils import ROOT


def download(cache):
    folder = cache / "cifar10_gray"
    folder.mkdir(parents=True, exist_ok=True)
    name, expected = next(iter(FILES["cifar10_gray"].items()))
    path = folder / name
    if path.exists():
        with path.open("rb") as file:
            if hashlib.file_digest(file, "md5").hexdigest() == expected:
                return path
    url = BASE["cifar10_gray"] + name
    size, chunk = 170052171, 2 * 1024 * 1024

    def read_range(start):
        end = min(size, start + chunk) - 1
        for attempt in range(3):
            try:
                with requests.get(url, headers={"Range": f"bytes={start}-{end}"}, params={"part": start},
                                  stream=True, timeout=(15, 40)) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                        raise ValueError(f"Unexpected HTTP range for bytes {start} to {end}")
                    data = response.content
                    if len(data) != end - start + 1:
                        raise ValueError(f"Incomplete range starting at byte {start}")
                    return data
            except requests.RequestException:
                if attempt == 2:
                    raise

    with NamedTemporaryFile(dir=folder, delete=False) as file:
        pending = Path(file.name)
    try:
        digest = hashlib.md5()
        with ThreadPoolExecutor(8) as pool, pending.open("wb") as file:
            for data in pool.map(read_range, range(0, size, chunk)):
                file.write(data)
                digest.update(data)
        if digest.hexdigest() != expected:
            raise ValueError("CIFAR-10 archive digest mismatch")
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / ".cache" / "external")
    args = parser.parse_args()
    print(download(args.cache))
