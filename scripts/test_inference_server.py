"""Smoke-test the furniture layout inference HTTP API."""

from __future__ import annotations

import json
from pathlib import Path
from urllib import request

import fire


def main(
    url: str = "http://127.0.0.1:5001/predict",
    input_json: str = "configs/infer/example_room.json",
) -> None:
    """Send one prediction request and print the structured response."""

    payload = json.loads(Path(input_json).read_text())
    body = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(http_request, timeout=120) as response:
        print(json.dumps(json.loads(response.read()), indent=2))


if __name__ == "__main__":
    fire.Fire(main)
