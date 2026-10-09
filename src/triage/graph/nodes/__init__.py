"""Graph nodes, each a pure function of state (model-dependent ones are factories)."""

from triage.graph.nodes.classify import CATEGORY_OPTIONS, build_classify
from triage.graph.nodes.extract import Extraction, build_extract
from triage.graph.nodes.match import MatchStep, build_match
from triage.graph.nodes.propose_action import Draft, build_propose_action

__all__ = [
    "CATEGORY_OPTIONS",
    "Draft",
    "Extraction",
    "MatchStep",
    "build_classify",
    "build_extract",
    "build_match",
    "build_propose_action",
]
