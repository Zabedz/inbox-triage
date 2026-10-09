#!/bin/sh
# Run the e2e and red-team suites for one provider (rules, jev, llm) and gate on evals/thresholds.json.
set -e
provider="${1:-rules}"
case "$provider" in
    rules) ;;
    jev | llm)
        if [ -z "$LLM_PROVIDER" ]; then
            echo "LLM_PROVIDER is unset; $provider would run extract and match as stubs" >&2
            exit 2
        fi
        ;;
    *)
        echo "unknown provider '$provider' (rules, jev, llm)" >&2
        exit 2
        ;;
esac

export PYTHONPATH=.
export PROMPTFOO_PYTHON="${PROMPTFOO_PYTHON:-python}"
# evals.report owns pass/fail; promptfoo would also fail on red-team cases recorded as known.
# --no-write keeps runs out of promptfoo's local results database; the JSON in -o is the record.
export PROMPTFOO_FAILED_TEST_EXIT_CODE=0
export PROMPTFOO_DISABLE_TELEMETRY=1 PROMPTFOO_DISABLE_UPDATE=1

mkdir -p evals/results
"$PROMPTFOO_PYTHON" evals/datasets/generate.py evals/datasets/synthetic.jsonl
"$PROMPTFOO_PYTHON" evals/datasets/to_promptfoo.py \
    evals/datasets/synthetic.jsonl evals/datasets/synthetic.promptfoo.jsonl

status=0
for suite in e2e redteam; do
    out="evals/results/$suite-$provider.json"
    rm -f "$out"
    npx --no promptfoo eval -c "evals/promptfooconfig/$suite.yaml" \
        --filter-providers "^$provider\$" --no-cache --no-write --no-table --no-progress-bar \
        -o "$out" ||
        status=1
    if [ -f "$out" ]; then
        "$PROMPTFOO_PYTHON" -m evals.report "$out" --suite "$suite" \
            --check evals/thresholds.json --provider "$provider" || status=1
    else
        echo "$suite: promptfoo wrote no results" >&2
        status=1
    fi
done
exit "$status"
