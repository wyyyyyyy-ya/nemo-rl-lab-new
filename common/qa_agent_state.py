"""QA Agent 状态机的纯 Python 辅助逻辑（不依赖 Ray / NeMo-RL）。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

SEARCH_ACTION_STOP_STRING = "</search>"
READ_ACTION_STOP_STRING = "</read>"
LITERAL_SEARCH_PLACEHOLDER = "检索词"

PROTOCOL_FABRICATED_RESULTS = "fabricated_results"
PROTOCOL_MIXED_ACTION = "mixed_action"
PROTOCOL_ANSWER_BEFORE_SEARCH = "answer_before_search"
PROTOCOL_REPEATED_QUERY = "repeated_query"
PROTOCOL_READ_BEFORE_SEARCH = "read_before_search"
PROTOCOL_UNKNOWN_READ = "unknown_read"
PROTOCOL_REPEATED_READ = "repeated_read"


def is_literal_search_placeholder(query: str) -> bool:
    """只识别模型把动作模板中的“检索词”原样当作真实 query。"""
    return " ".join(str(query).strip().lower().split()) == LITERAL_SEARCH_PLACEHOLDER


def last_assistant_text(message_log: Sequence[Mapping[str, Any]]) -> str:
    """仅返回当前轮（最后一条）assistant 文本，避免历史动作干扰状态解析。"""
    for message in reversed(message_log):
        if message.get("role") == "assistant":
            return str(message.get("content", "")).strip()
    return ""


def credited_search_hit(
    *,
    results_found: bool,
    used_fallback: bool,
    fallback_counts_as_hit: bool,
) -> bool:
    """只有模型原始 query 命中时才默认记功；fallback 可通过配置恢复旧行为。"""
    return results_found and (not used_fallback or fallback_counts_as_hit)


def protocol_violation(
    *,
    has_fabricated_results: bool,
    has_search: bool,
    has_answer: bool,
    search_count: int,
    search_query: str = "",
    last_search_query: str = "",
    has_read: bool = False,
    read_ref: str = "",
    allowed_read_refs: Sequence[str] = (),
    read_history: Sequence[str] = (),
) -> str | None:
    """Return the first hard protocol violation for the current assistant turn.

    The ordering is intentional: fabricated environment output is always the
    strongest violation, followed by mutually exclusive action violations.
    """
    if has_fabricated_results:
        return PROTOCOL_FABRICATED_RESULTS
    if sum((has_search, has_read, has_answer)) > 1:
        return PROTOCOL_MIXED_ACTION
    if has_answer and search_count <= 0:
        return PROTOCOL_ANSWER_BEFORE_SEARCH
    if has_read and search_count <= 0:
        return PROTOCOL_READ_BEFORE_SEARCH
    if has_read and read_ref not in allowed_read_refs:
        return PROTOCOL_UNKNOWN_READ
    if has_read and read_ref in read_history:
        return PROTOCOL_REPEATED_READ
    if (
        has_search
        and last_search_query.strip()
        and " ".join(search_query.lower().split())
        == " ".join(last_search_query.lower().split())
    ):
        return PROTOCOL_REPEATED_QUERY
    return None


def fallback_eligible(
    *,
    enabled: bool,
    has_prior_search_hit: bool,
    search_count_after_current: int,
    fallback_after_failed_searches: int,
) -> bool:
    """只有自主搜索达到失败阈值且此前从未命中时，才允许环境执行一次 fallback。"""
    return (
        enabled
        and not has_prior_search_hit
        and search_count_after_current >= fallback_after_failed_searches
    )


def next_action_stop_strings(terminateds: Sequence[bool]) -> list[list[str] | None]:
    """为仍在运行的 episode 保留工具动作 stop，结束样本不再设置。"""
    return [
        None if terminated else [SEARCH_ACTION_STOP_STRING, READ_ACTION_STOP_STRING]
        for terminated in terminateds
    ]
