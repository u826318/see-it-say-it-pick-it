from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Iterable

import httpx


def download_file(
    url: str,
    destination: Path,
    *,
    expected_size: int,
    token: str | None,
    chunk_size: int = 64 * 1024 * 1024,
    max_retries: int = 5,
) -> None:
    """Download one file with bounded HTTP range requests and resume support."""
    destination = Path(destination)
    partial = Path(f"{destination}.part")
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        actual_size = destination.stat().st_size
        if actual_size == expected_size:
            return
        raise RuntimeError(
            f"Refusing to replace {destination}: expected {expected_size} bytes, "
            f"found {actual_size}."
        )

    if partial.exists() and partial.stat().st_size > expected_size:
        raise RuntimeError(f"Partial file is larger than expected: {partial}")

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    timeout = httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=30.0)

    with httpx.Client(follow_redirects=True, timeout=timeout) as client:
        consecutive_failures = 0
        while True:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset == expected_size:
                os.replace(partial, destination)
                return

            end = min(offset + chunk_size, expected_size) - 1
            request_headers = {**headers, "Range": f"bytes={offset}-{end}"}
            before = offset
            try:
                with client.stream("GET", url, headers=request_headers) as response:
                    if response.status_code != 206:
                        raise RuntimeError(
                            f"Expected HTTP 206 for byte range {offset}-{end}, "
                            f"received {response.status_code}."
                        )
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {offset}-{end}/"):
                        raise RuntimeError(f"Unexpected Content-Range: {content_range!r}")
                    with partial.open("ab") as output:
                        for block in response.iter_bytes(chunk_size=1024 * 1024):
                            output.write(block)

                after = partial.stat().st_size
                if after != end + 1:
                    raise RuntimeError(
                        f"Range ended at byte {after - 1}; expected byte {end}."
                    )
                consecutive_failures = 0
            except (httpx.HTTPError, OSError, RuntimeError):
                after = partial.stat().st_size if partial.exists() else 0
                consecutive_failures = 0 if after > before else consecutive_failures + 1
                if consecutive_failures >= max_retries:
                    raise
                time.sleep(min(2**consecutive_failures, 15))


def download_snapshot(
    repo_id: str,
    revision: str,
    local_dir: Path,
    *,
    include: Iterable[str] | None = None,
) -> None:
    try:
        from huggingface_hub import HfApi, get_token, hf_hub_url
    except ImportError as exc:
        raise RuntimeError(
            "Install huggingface-hub first: python -m pip install huggingface-hub"
        ) from exc

    selected = set(include or [])
    info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
    files = [
        item
        for item in info.siblings
        if not selected or item.rfilename in selected
    ]
    missing = selected.difference(item.rfilename for item in files)
    if missing:
        raise RuntimeError(f"Files not found in repository: {sorted(missing)}")

    token = get_token()
    for index, item in enumerate(files, start=1):
        if item.size is None:
            raise RuntimeError(f"Repository did not report a size for {item.rfilename}")
        destination = Path(local_dir) / item.rfilename
        print(f"[{index}/{len(files)}] {item.rfilename} ({item.size:,} bytes)", flush=True)
        download_file(
            hf_hub_url(repo_id, item.rfilename, revision=revision),
            destination,
            expected_size=item.size,
            token=token,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable Hugging Face snapshot downloader")
    parser.add_argument("repo_id")
    parser.add_argument("revision")
    parser.add_argument("local_dir", type=Path)
    parser.add_argument("--include", action="append", default=[])
    args = parser.parse_args()
    download_snapshot(
        args.repo_id,
        args.revision,
        args.local_dir,
        include=args.include,
    )


if __name__ == "__main__":
    main()
