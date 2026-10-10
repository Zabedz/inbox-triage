#!/bin/sh
# e2e on the real labelled set (data/labels.jsonl), then thresholds fitted before SPLIT_DATE and
# scored after it. Everything written stays in data/; the models see redacted text only.
set -e
export PYTHONPATH=.
export PROMPTFOO_PYTHON="${PROMPTFOO_PYTHON:-python}"
# Validate before the paid run; the shell cannot reject 2026-02-30.
"$PROMPTFOO_PYTHON" -c 'import sys, datetime; datetime.date.fromisoformat(sys.argv[1])' \
    "${SPLIT_DATE:-}" 2>/dev/null || {
    echo "set SPLIT_DATE to a real date, YYYY-MM-DD" >&2
    exit 2
}
export PROMPTFOO_FAILED_TEST_EXIT_CODE=0
export PROMPTFOO_DISABLE_TELEMETRY=1 PROMPTFOO_DISABLE_UPDATE=1

"$PROMPTFOO_PYTHON" evals/datasets/to_promptfoo.py data/labels.jsonl data/labels.promptfoo.jsonl
npx --no promptfoo eval -c evals/promptfooconfig/e2e-real.yaml --no-cache --no-write \
    --no-table --no-progress-bar -o data/results-real.json
"$PROMPTFOO_PYTHON" -m evals.report data/results-real.json --suite e2e
"$PROMPTFOO_PYTHON" -m evals.calibrate data/results-real.json --split-date "$SPLIT_DATE"
