"""Tiny language heuristic shared by the agent's output validator and the evaluator.

Share of CJK characters among letters, ignoring citations and .md file paths (which are
Latin even inside a Chinese answer). > 0.3 reads as Chinese, < 0.3 as English.
"""

from __future__ import annotations

import re

CITATION = re.compile(r"[\[【][^\]】]*\.md[^\]】]*[\]】]")  # [x.md] or full-width 【x.md】
MD_PATH = re.compile(r"[\w./-]+\.md")
CJK = re.compile(r"[一-鿿]")
LATIN = re.compile(r"[A-Za-z]")


def cjk_share(text: str) -> float:
    text = MD_PATH.sub("", CITATION.sub("", text))
    cjk, latin = len(CJK.findall(text)), len(LATIN.findall(text))
    return cjk / (cjk + latin) if cjk + latin else 0.0


def is_chinese(text: str) -> bool:
    return cjk_share(text) > 0.3
