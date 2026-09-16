from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import requests

from kagglesdk.kaggle_http_client import KaggleHttpClient
from kagglesdk.competitions.services.competition_api_service import CompetitionApiClient
from kagglesdk.competitions.types.competition_api_service import (
    ApiCreateSubmissionRequest,
    ApiListSubmissionsRequest,
    ApiStartSubmissionUploadRequest,
)


def client_from_token(token_path: Path) -> CompetitionApiClient:
    token = token_path.expanduser().read_text().strip()
    if not token:
        raise RuntimeError(f"Empty Kaggle API token: {token_path}")
    os.environ["KAGGLE_API_TOKEN"] = token
    return CompetitionApiClient(KaggleHttpClient())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--message", required=True)
    parser.add_argument(
        "--competition",
        default="hyperspectral-object-detection-challenge-2026",
    )
    parser.add_argument(
        "--token",
        type=Path,
        default=Path.home() / ".kaggle" / "access_token",
    )
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument(
        "--curl-upload",
        action="store_true",
        help="Upload the blob with curl instead of requests (useful when requests cannot reach Google's upload host).",
    )
    parser.add_argument(
        "--upload-resolve-ip",
        default=None,
        help="Optional IPv4 address to force for www.googleapis.com via curl --resolve.",
    )
    args = parser.parse_args()

    client = client_from_token(args.token)

    if args.list_only:
        req = ApiListSubmissionsRequest()
        req.competition_name = args.competition
        response = client.list_submissions(req)
        for sub in response.submissions:
            print(sub.ref, sub.status, sub.public_score, sub.description)
        return

    if not args.file.is_file():
        raise FileNotFoundError(args.file)

    stat = args.file.stat()
    start = ApiStartSubmissionUploadRequest()
    start.competition_name = args.competition
    start.file_name = args.file.name
    start.content_length = stat.st_size
    start.last_modified_epoch_seconds = int(stat.st_mtime)
    upload = client.start_submission_upload(start)
    if not upload.create_url or not upload.token:
        raise RuntimeError("Kaggle did not return an upload URL/token")

    if args.curl_upload:
        cmd = [
            "curl",
            "-4",
            "-fsS",
            "--retry",
            "5",
            "--retry-all-errors",
            "--connect-timeout",
            "15",
            "--max-time",
            "180",
            "--request",
            "PUT",
            "--upload-file",
            str(args.file),
        ]
        if args.upload_resolve_ip:
            cmd += ["--resolve", f"www.googleapis.com:443:{args.upload_resolve_ip}"]
        cmd.append(upload.create_url)
        subprocess.run(cmd, check=True)
    else:
        with args.file.open("rb") as f:
            response = requests.put(upload.create_url, data=f, timeout=180)
        response.raise_for_status()

    create = ApiCreateSubmissionRequest()
    create.competition_name = args.competition
    create.blob_file_tokens = upload.token
    create.submission_description = args.message
    result = client.create_submission(create)
    print(f"submitted ref={result.ref} message={result.message}")


if __name__ == "__main__":
    main()
