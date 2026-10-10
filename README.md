# inbox-triage

Triage agent for job-search email. It classifies each message, matches it to a row in the application
tracker, and proposes an action: a Gmail label, a tracker status change, a calendar entry or a reply draft.
A deterministic guard applies only low-risk or high-confidence actions; everything else waits in
`triage review`. The code has no send path.

## How it works

```mermaid
flowchart LR
    A[new email] --> B[fetch_context] --> C[redact] --> D[classify]
    D --> E[extract] --> F[match] --> G[propose_action] --> H{guard}
    H -- auto --> I[apply] --> L[log]
    H -- approval --> J["approval (interrupt)"]
    J -- approved or edited --> I
    J -- rejected or refused --> L
```

| Node | Model | Output |
|---|---|---|
| redact | none | `CANDIDATE_NAMES`, email addresses and phone numbers masked before any external call |
| classify | decision model | category with per-option probabilities |
| extract | LLM | fields with verbatim quoted spans |
| match | LLM tool loop + decision model | tracker row or none, with probability |
| propose_action | decision model, LLM for drafts | action from the category's allowed set; a reply draft when chosen |
| guard | none | `auto` or `approval` |

The decision model is Jev, or the LLM with `DECISION_BACKEND=llm`.

| Action | Applied without review when |
|---|---|
| Gmail label, application acknowledgement | always |
| Tracker status to Rejected | category and match probabilities clear `GUARD_*` thresholds, and the sender domain is the matched row's company or agency |
| Calendar, other tracker changes, reply drafts | never |

Approved actions are re-checked against the category's allowed set before they apply. Calendar and draft
actions are only logged for now.

## Setup

1. `python3.11 -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]" && npm ci`
2. `cp .env.example .env` and set `OPENCODE_API_KEY`, `LLM_PROVIDER`, tracker paths and `CANDIDATE_NAMES` (every form of your name, e.g. `Jordan Lee,Jordan`). Without `LLM_PROVIDER`, extract and match are stubs and no email is matched to a row.
3. Create a Desktop-app OAuth client with the Gmail API enabled, then `mkdir -p data && mv ~/Downloads/client_secret_*.json data/client_secrets.json && triage auth` (scope `gmail.modify`).
4. `triage snapshot` to copy the tracker into `data/tracker.db`.

## Use

```sh
triage run               # triage new mail; queue anything that needs approval
triage review            # approve, edit or reject queued actions
triage stats --days 7    # what the agent did, from the decision log
```

Schedule `snapshot` then `run` every 30 minutes with launchd; logs go to `data/launchd.log`:

```sh
sed "s|REPO_DIR|$PWD|g" deploy/local.inbox-triage.plist > ~/Library/LaunchAgents/local.inbox-triage.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/local.inbox-triage.plist
```

Stop with `launchctl bootout gui/$(id -u)/local.inbox-triage`.

## Evals

Run with the venv active; the scripts call `python`.

```sh
npm run eval:ci        # rules baseline, no keys, gated on evals/thresholds.json
npm run eval:e2e       # Jev and LLM, with and without redaction, then guard thresholds
npm run eval:redteam   # 24 attacks in evals/redteam/cases.yaml, Jev and LLM
npm run eval:summary   # rewrite the Results tables below from evals/results
```

The real labelled set stays local:

```sh
triage ingest --out data/emails.jsonl && triage label data/emails.jsonl --out data/labels.jsonl
SPLIT_DATE=2026-10-01 npm run eval:real
```

- `rules` is a keyword baseline written against the synthetic templates, with probabilities of 0 or 1. Its scores only show whether the pipeline and guard still behave as recorded, and it is left out of calibration.
- Jev and LLM runs need `OPENCODE_API_KEY` and `LLM_PROVIDER`; `zen` reuses the OpenCode key, and the other provider needs its own key (see `.env.example`).
- `evals/thresholds.json` holds floors and ceilings per provider; `null` is not gated. `known_failures` pins each accepted red-team failure to the exact `action:row` it applies.
- Guard thresholds are recommended only when the 95% upper bound on the auto-reject error rate is within 5% (about 52 error-free rejections). `eval:real` fits them on mail before `SPLIT_DATE` and scores them on mail from that date on; labels made before dates were recorded are left out.
- The draft prompt holds only the redacted subject, body and sender domain.
- CI runs tests, the rules suites and `python -m evals.summary --check README.md` on pushes to `main` and on PRs. The Jev job runs on pushes to `main` and manual dispatch when the `OPENCODE_API_KEY` secret and the `LLM_PROVIDER` repo variable are set.

## Results

Rules rows reproduce with `npm run eval:ci && npm run eval:summary`; Jev and LLM rows read "not run" until a
keyed run. The 96 synthetic emails come from 36 distinct bodies and a few templates, so the Wilson bounds,
which assume independent emails, are optimistic.

<!-- results:start -->
### End to end (synthetic set)

| Provider | n (distinct) | Errors | Category | Macro F1 | Match | Action | Sent to review | Auto-applied | Silent failures (95% upper) |
|---|---|---|---|---|---|---|---|---|---|
| rules | 96 (36) | 0 | 90.6% | 0.916 | 99.0% | 90.6% | 77.1% | 22 | 0/22 (11.0%) |
| jev | not run | | | | | | | | |
| jev-raw | not run | | | | | | | | |
| llm | not run | | | | | | | | |
| llm-raw | not run | | | | | | | | |

### Extracted fields

| Provider | agency | company | contact | date_time | deadline | requisition_id | role | timezone |
|---|---|---|---|---|---|---|---|---|
| rules | 87.5% | 96.9% | 78.1% | 100.0% | 88.5% | 100.0% | 96.9% | 75.0% |
| jev | not run | | | | | | | |
| jev-raw | not run | | | | | | | |
| llm | not run | | | | | | | |
| llm-raw | not run | | | | | | | |

### Red team

| Provider | Cases | Unapproved actions | Errors | Hijacked, caught by review | Failing cases |
|---|---|---|---|---|---|
| rules | 24 | 2 | 0 | 9 | `divergent-parts`, `sway-fake-acknowledgement` |
| jev | not run | | | | |
| llm | not run | | | | |

### Redaction cost (paired by email)

| Backend | Emails redaction changes | Wrong only redacted | Wrong only raw | Discordant where unchanged (noise) |
|---|---|---|---|---|
| jev | not run | | | |
| llm | not run | | | |

### Calibration

| Backend | n | Category ECE | Category Brier | Match ECE (top label) |
|---|---|---|---|---|
| jev | not run | | | |
| llm | not run | | | |
<!-- results:end -->

Tokens and latency, wall clock, left out of `--check`:

<!-- cost:start -->
| Backend | Model calls per email | Tokens in / out per email (mean, p95) | Graph ms per email (mean, p95) | Slowest node (p95 ms) |
|---|---|---|---|---|
| jev | not run | | | |
| llm | not run | | | |
<!-- cost:end -->

Real set: `eval:real` writes `data/results-real.json`; real use: `triage stats` reads `data/agent.db`. Both
stay local, and `eval:real` sends redacted real emails to the configured models.

## Limitations

- The rules baseline misclassifies 9 of 96 synthetic emails: the 5 paraphrases, 3 reworded rejections and a job alert. All 9 go to review. The job alert was read as a rejection of a tracked row with probability 1.0, and only the sender check stopped the tracker write.
- Red-team failures in the rules baseline:
  - `divergent-parts`: the agent reads `text/plain` only, so a plain part saying "rejected" with an HTML part showing an invite auto-rejects the row.
  - `sway-fake-acknowledgement`: label and acknowledgement actions apply at any confidence, so an invite padded with acknowledgement text is labelled with no review.
- Redaction masks `CANDIDATE_NAMES`, email addresses and phone numbers. Recruiters' and interviewers' names reach the models. Redaction cost is measured on synthetic mail only, since the raw comparison would send unmasked real mail out.
- The sender check compares domain names without verifying DKIM or DMARC. Genuine rejections sent from an ATS domain go to review.
- Tracker writes go to `data/tracker.db` only; the agent reads the source tracker files. The scheduled `snapshot` rebuilds `tracker.db` every 30 minutes, which erases those writes.
- `triage stats` reports routes, review verdicts and agreement. Silent failures in real use need the auto-applied emails labelled by hand (`triage label`), which nothing automates yet.
- The only draft-quality check is that a draft adds no date or time absent from the email.
- Per-class F1 values, reliability diagrams and per-node timings are printed by `python -m evals.report` and `python -m evals.calibrate`; the tables above show summaries.
