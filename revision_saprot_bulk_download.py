"""Download and verify byte ranges of the official AlphaFold human proteome archive."""

import argparse
import concurrent.futures
import re
import shutil
import subprocess
from pathlib import Path


URL = "https://ftp.ebi.ac.uk/pub/databases/alphafold/v6/UP000005640_9606_HUMAN_v6.tar"
SIZE = 5177506304
CHUNK = 32 * 1024 * 1024


def fetch(directory, start):
    end = min(SIZE, start + CHUNK) - 1
    path = directory / f"{start:011d}.bin"
    headers = path.with_suffix(".headers")
    expected = end - start + 1
    if path.exists() and path.stat().st_size == expected and headers.exists():
        prior = headers.read_text(encoding="ascii", errors="replace")
        if f"content-range: bytes {start}-{end}/{SIZE}" in prior.lower():
            return path
    subprocess.run(["curl.exe", "-fLsS", "--retry", "2", "--max-time", "180",
                    "--range", f"{start}-{end}", "-D", str(headers), URL, "-o", str(path)], check=True)
    header = headers.read_text(encoding="ascii", errors="replace")
    ranges = re.findall(r"(?im)^content-range: bytes (\d+)-(\d+)/(\d+)\s*$", header)
    if not ranges or tuple(map(int, ranges[-1])) != (start, end, SIZE) or path.stat().st_size != expected:
        raise ValueError(f"Incorrect response for bytes {start}-{end}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new output path")
    directory = args.output.parent / "afdb_v6_chunks"
    directory.mkdir(parents=True, exist_ok=True)
    offsets = list(range(0, SIZE, CHUNK))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending = [executor.submit(fetch, directory, start) for start in offsets]
        for i, future in enumerate(concurrent.futures.as_completed(pending), 1):
            future.result()
            if i <= 3 or i % 10 == 0 or i == len(pending):
                print("Verified byte ranges", i, "/", len(pending), flush=True)
    with args.output.open("xb") as output:
        for future in pending:
            with future.result().open("rb") as stream:
                shutil.copyfileobj(stream, output, 4 * 1024 * 1024)
    if args.output.stat().st_size != SIZE:
        raise ValueError("Assembled archive has an incorrect length")
    print("Verified complete archive", args.output.stat().st_size, flush=True)


if __name__ == "__main__":
    main()
