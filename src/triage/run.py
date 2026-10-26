"""`triage run`: triage each new email through the agent graph."""

from __future__ import annotations

import fcntl
import logging
from typing import Any

from triage.agent import (
    EMAIL_ERRORS,
    is_transient,
    open_checkpointer,
    open_decision_log,
    private_dir,
    production_graph,
)
from triage.config import Config
from triage.gmail.client import GmailClient
from triage.gmail.ingest import build_service
from triage.storage import DecisionLog, PendingApprovals
from triage.tracker.snapshot import snapshot

logger = logging.getLogger("triage.run")

LOCK_FILE = "run.lock"


def run(
    cfg: Config,
    graph: Any = None,
    client: GmailClient | None = None,
    pending: PendingApprovals | None = None,
    decision_log: DecisionLog | None = None,
) -> list[str]:
    """Triage each new email; return the thread ids left awaiting approval.

    Emails already decided, pending, or failed too often are skipped, so the schedule
    can rerun the same Gmail query. An outage stops the run without counting against
    any email. An overlapping run exits at once.
    """
    private_dir(cfg.tracker.data_dir)
    with (cfg.tracker.data_dir / LOCK_FILE).open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            logger.warning("another triage run holds %s; exiting", lock.name)
            return []
        return _run(cfg, graph, client, pending, decision_log)


def _run(
    cfg: Config,
    graph: Any,
    client: GmailClient | None,
    pending: PendingApprovals | None,
    decision_log: DecisionLog | None,
) -> list[str]:
    pending_store = pending or PendingApprovals(cfg.agent_db)
    log_store = decision_log or open_decision_log(cfg)
    if graph is None or client is None:
        # Re-read the source tracker so edits made outside the agent are seen.
        snapshot(cfg)
        service = build_service(cfg, interactive=False)
        client = client or GmailClient(service)
        graph = graph or production_graph(
            cfg, service, log_store, open_checkpointer(cfg.checkpoints_db)
        )

    pending_threads: list[str] = []
    for message_id in client.list_job_messages(cfg.gmail.query):
        if log_store.is_done(message_id) or pending_store.has(message_id):
            continue
        thread_id = f"email-{message_id}"
        config = {"configurable": {"thread_id": thread_id}}
        try:
            email = client.get_email(message_id)
            result = graph.invoke({"email_id": message_id, "email": email}, config=config)
        except EMAIL_ERRORS as exc:
            if is_transient(exc):
                logger.warning("stopping run at %s: %s", message_id, type(exc).__name__)
                break
            logger.warning("triage failed for %s: %s", message_id, type(exc).__name__)
            log_store.log_error(message_id, type(exc).__name__)
            if graph.checkpointer is not None:
                graph.checkpointer.delete_thread(thread_id)
            continue
        interrupts = result.get("__interrupt__") or []
        for interrupt in interrupts:
            pending_store.add(thread_id, message_id, interrupt.value)
            pending_threads.append(thread_id)
        if not interrupts and graph.checkpointer is not None:
            graph.checkpointer.delete_thread(thread_id)

    logger.info("run complete: %d awaiting approval", len(pending_threads))
    return pending_threads
