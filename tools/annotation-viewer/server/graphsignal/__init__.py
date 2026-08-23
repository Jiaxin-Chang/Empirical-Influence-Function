"""Vendored GraphSignal annotation (tree-sitter + optional LLM).

Copied from code-corr-annotation ``src/annotate/`` so the EIF repo does not
depend on an external checkout. LLM settings come from repo-root ``eif_api.env``.
"""

from .run import annotate_row

__all__ = ["annotate_row"]
