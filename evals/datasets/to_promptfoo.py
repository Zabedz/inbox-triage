"""Convert a labelled JSONL into a promptfoo test JSONL (fields wrapped in ``vars``)."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def convert(src: str | Path, dst: str | Path) -> int:
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with dst.open("w") as out:
        for line in src.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            out.write(
                json.dumps(
                    {
                        "vars": {
                            "email_id": record["email_id"],
                            "from_": record["from_"],
                            "subject": record["subject"],
                            "body": record["body"],
                            "category": record["category"],
                        },
                        "description": record["email_id"],
                    }
                )
                + "\n"
            )
            count += 1
    return count


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    print(f"wrote {convert(src, dst)} test cases to {dst}")
