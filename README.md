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
