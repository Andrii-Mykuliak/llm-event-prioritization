"""Response parsing: identical to the rule used for the reference results.

parse_labels():   scan the response (any <think> block removed) for tokens of two or more capital letters, keep the ones
                  that equal an identifier of the current presentation, in order of first appearance and without
                  duplicates, and take the first k.
format_ok:        three such identifiers were found.
invalid_labels(): tokens of exactly three capital letters that are not identifiers of the presentation ("invented IDs").
"""

from __future__ import annotations

import re

_THINK = re.compile(r"<think>.*?</think>", re.S)


def invalid_labels(text: str, valid: set[str], length: int = 3) -> list[str]:
    text = _THINK.sub("", text)
    return [t for t in re.findall(r"\b[A-Z]{%d}\b" % length, text) if t not in valid]


def parse_labels(text: str, valid: set[str], k: int = 3) -> list[str]:
    text = _THINK.sub("", text)
    found: list[str] = []
    for tok in re.findall(r"[A-Z]{2,}", text.upper()):
        if tok in valid and tok not in found:
            found.append(tok)
    return found[:k]
