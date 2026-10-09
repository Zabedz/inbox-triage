# inbox-triage

Job-search inbox triage agent. Classifies email about job applications, matches it to the application
tracker, and proposes an action (tracker status change, calendar entry, reply draft). Anything uncertain
waits for approval. Under construction.

## Schedule

Runs `triage snapshot` then `triage run` every 30 minutes. Run `triage auth` once first (label scope).

```sh
sed "s|REPO_DIR|$PWD|g" deploy/local.inbox-triage.plist > ~/Library/LaunchAgents/local.inbox-triage.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/local.inbox-triage.plist
```

Review queued actions with `triage review`. Stop with `launchctl bootout gui/$(id -u)/local.inbox-triage`.

## Evals

Run with the venv active (scripts call `python`).

```sh
npm run eval:ci        # rules baseline, no keys, gated on evals/thresholds.json
npm run eval:e2e       # full graph: rules vs Jev vs LLM, then recommended guard thresholds
npm run eval:redteam   # injection, extraction and spoofing cases in evals/redteam/cases.yaml
```

- `rules` is a keyword baseline written against the synthetic templates. Its scores only show whether the pipeline and guard still behave as recorded.
- Jev and LLM runs need `OPENCODE_API_KEY`, `LLM_PROVIDER`, `LLM_MODEL` and that provider's key (see `.env.example`).
- `evals/thresholds.json` holds floors and ceilings per provider; `null` is not gated. `known_failures` pins each accepted red-team failure to the exact `action:row` it applies; any other unapproved action fails.
- `eval:e2e` prints `GUARD_*` lines only when the 95% upper bound on the auto-reject error rate is within 5% (about 52 error-free rejections); paste them into `.env`.
- Drafts are written from the redacted email alone; tracker rows never reach the draft prompt.
- CI (`.github/workflows/ci.yml`): tests and rules suites on pushes to `main` and on PRs. The Jev job runs on `main` and manual dispatch when the `OPENCODE_API_KEY` secret is set; set repo variables `LLM_PROVIDER` and `LLM_MODEL`.
