"""Graph nodes, each a pure function of state (model-dependent ones are factories)."""

from triage.graph.nodes.classify import CATEGORY_OPTIONS, build_classify

__all__ = ["CATEGORY_OPTIONS", "build_classify"]
