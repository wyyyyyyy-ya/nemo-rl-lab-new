"""多轮 QA 文档检索 Agent 环境（NeMo-RL 0.6.0）。

交互协议（考试要求）：
  1. 模型输出 <search>关键词</search> → 环境在 docs_dir 检索 markdown，回灌 [检索结果]
  2. 可多次检索（受 max_searches 限制）
  3. 模型输出 \\boxed{答案} → 调用 common/rewards 判分，episode 结束

与 QARewardEnv 的区别：多轮、支持检索工具；最终判分逻辑与单轮 QA 一致。
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Optional, TypedDict

from common.doc_search import (
    DocumentSearchIndex,
    is_low_quality_snippet,
    search_result_bodies,
)
from common.question_bank import QuestionBankIndex
from common.qa_agent_state import (
    PROTOCOL_ANSWER_BEFORE_SEARCH,
    PROTOCOL_FABRICATED_RESULTS,
    PROTOCOL_MIXED_ACTION,
    PROTOCOL_REPEATED_QUERY,
    PROTOCOL_READ_BEFORE_SEARCH,
    PROTOCOL_REPEATED_READ,
    PROTOCOL_UNKNOWN_READ,
    credited_search_hit as _credited_search_hit,
    fallback_eligible as _fallback_eligible,
    last_assistant_text as _last_assistant_text,
    next_action_stop_strings as _next_action_stop_strings,
    protocol_violation as _protocol_violation,
    is_literal_search_placeholder as _is_literal_search_placeholder,
)
from common.rewards.qa_reward import extract_boxed

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_SEARCH_OPEN_TAG = re.compile(r"<search\s*>", re.IGNORECASE)
_SEARCH_CLOSE_TAG = re.compile(r"</search\s*>", re.IGNORECASE)
_READ_TAG = re.compile(r"<read>\s*(D\d+(?::C\d+)?)\s*</read>", re.IGNORECASE)
_FABRICATED_RESULTS_TAG = re.compile(r"[\[【]\s*检索结果\s*[\]】]")


class QAAgentMetadata(TypedDict, total=False):
    query: str
    expected_answer: str
    search_count: int
    last_search_query: str
    has_search_hit: bool
    fallback_count: int
    read_count: int
    allowed_read_refs: list[str]
    read_history: list[str]
    read_ref_queries: dict[str, str]
    read_ref_evidence_scores: dict[str, float]
    read_ref_incomplete: dict[str, bool]
    last_search_candidate_only: bool
    last_search_read_refs: list[str]
    has_answer_evidence: bool
    # pending_valid_read_bonus: bool  # 暂停 read 延迟奖励时不记录。


_FILL_BLANK_PLACEHOLDER = re.compile(r"【\d+】")
_SUGGEST_SKIP_TOKENS = frozenset(
    {"根据", "下面", "一道", "填空题", "选择题", "单选题", "多选题", "两种", "三种", "四种", "题目"}
)
_PLACEHOLDER_SEARCH_NORMALIZED = frozenset(
    {
        "题干关键词",
        "关键词",
        "专业名词",
        "从题目提取的专业名词",
        "题目原文或专业名词",
        "题干",
        "search",
        "检索",
    }
)


def _compact_search_suggest(text: str, max_len: int = 30) -> str:
    """把题面压成适合 grep 的短检索词（去填空占位符、去套话）。"""
    text = _FILL_BLANK_PLACEHOLDER.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_len:
        return text
    tokens = re.findall(r"[A-Za-z]{2,}|[\u4e00-\u9fff]{2,}", text)
    buf = ""
    for t in tokens:
        if t in _SUGGEST_SKIP_TOKENS:
            continue
        candidate = f"{buf} {t}".strip() if buf else t
        if len(candidate) > max_len:
            break
        buf = candidate
    return buf or text[:max_len]


def _suggest_search_from_query(query: str) -> str:
    """从题面抽取更具体的检索建议。"""
    m = re.search(r"题目[：:](.*?)(?:\n\n选项|\n选项：|\n选项:|\Z)", query, re.DOTALL)
    if m:
        return _compact_search_suggest(m.group(1).strip())
    m2 = re.search(r"题目[：:](.+)", query)
    if m2:
        return _compact_search_suggest(m2.group(1).strip())
    return _compact_search_suggest(query.strip())


def _is_placeholder_search(query: str) -> bool:
    """检测模型是否把 prompt 占位符原样当作 search 内容。"""
    return _normalize_search_query(query) in _PLACEHOLDER_SEARCH_NORMALIZED


def _normalize_search_query(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


def _is_short_search(query: str, min_len: int) -> bool:
    compact = re.sub(r"\s+", "", query)
    return len(compact) < min_len


def _all_assistant_text(message_log: LLMMessageLogType) -> str:
    parts = [
        str(msg.get("content", "")).strip()
        for msg in message_log
        if msg.get("role") == "assistant" and str(msg.get("content", "")).strip()
    ]
    return "\n".join(parts)


def _parse_search_query(text: str) -> str | None:
    """提取最后一个闭合标签之前距离最近的 search 开标签内容。"""
    close_matches = list(_SEARCH_CLOSE_TAG.finditer(text))
    if not close_matches:
        return None
    close_match = close_matches[-1]
    open_matches = list(_SEARCH_OPEN_TAG.finditer(text, 0, close_match.start()))
    if not open_matches:
        return None
    open_match = open_matches[-1]
    query = text[open_match.end() : close_match.start()].strip()
    return query or None


def _parse_read_ref(text: str) -> str | None:
    match = _READ_TAG.search(text)
    return match.group(1).upper() if match else None


def _extract_option_terms(query: str) -> list[str]:
    """从题面选项行提取文本，用于检测 search 是否带入选项名称。"""
    terms: list[str] = []
    for m in re.finditer(r"^[A-L][\.、．\)]\s*(.+)$", query, re.MULTILINE):
        text = re.sub(r"\s+", " ", m.group(1).strip())
        if len(text) >= 4:
            terms.append(text)
    return terms


def _evidence_norm(text: str) -> str:
    return "".join(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", text.lower()))


def _expected_evidence_terms(query: str, expected: str) -> tuple[str, list[str], set[str]]:
    """Extract answer-bearing phrases without exposing them to the agent."""
    match = re.match(r"\s*\[(\w+)\]\s*(.*)", expected, flags=re.DOTALL)
    if not match:
        return "", [expected.strip()] if expected.strip() else [], set()
    qtype, gold = match.group(1).lower(), match.group(2).strip()
    if qtype in {"single", "multiple", "bool"}:
        gold_letters = set(re.findall(r"[A-Z]", gold.upper()))
        options = {
            option.group(1).upper(): option.group(2).strip()
            for option in re.finditer(
                r"^([A-L])[\.、．\)]\s*(.+)$", query, flags=re.MULTILINE
            )
        }
        return qtype, [options[letter] for letter in sorted(gold_letters) if letter in options], gold_letters
    if qtype in {"fill", "short"}:
        terms: list[str] = []
        for item in gold.split("|||"):
            alternatives = [part.strip() for part in re.split(r"[/|]", item) if part.strip()]
            if alternatives:
                terms.extend(alternatives)
        return qtype, terms, set()
    return qtype, [gold] if gold else [], set()


def _answer_evidence_score(query: str, expected: str, text: str) -> float:
    """Return 0..1 coverage of expected answer phrases for valid-read shaping."""
    _, terms, _ = _expected_evidence_terms(query, expected)
    if not terms:
        return 0.0
    normalized = _evidence_norm(text)
    hits = sum(
        1
        for term in terms
        if (term_norm := _evidence_norm(term)) and term_norm in normalized
    )
    return hits / len(terms)


def _search_biased_by_options(search_query: str, query: str) -> str | None:
    sq = _normalize_search_query(search_query)
    for opt in _extract_option_terms(query):
        if _normalize_search_query(opt) in sq:
            return opt
    return None


def _split_search_queries(keyword: str, max_len: int = 30) -> list[str]:
    """长检索词拆成多个短 query。"""
    keyword = keyword.strip()
    if len(keyword) <= max_len:
        return [keyword]
    chunks = [c for c in re.split(r"[\s,，、；;]+", keyword) if c.strip()]
    parts: list[str] = []
    buf = ""
    for c in chunks:
        if len(buf) + len(c) + 1 <= max_len:
            buf = f"{buf} {c}".strip() if buf else c
        else:
            if buf:
                parts.append(buf)
            buf = c
    if buf:
        parts.append(buf)
    for c in chunks:
        if len(c) >= 4 and c not in parts:
            parts.append(c)
    return parts[:3] or [keyword[:max_len]]


def _is_low_quality_snippet(snippet: str) -> bool:
    return is_low_quality_snippet(snippet)


def _merge_search_results(
    *groups: list[tuple[str, str]],
    top_k: int,
) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    merged: list[tuple[str, str]] = []
    for group in groups:
        for rel, snippet in group:
            key = (rel, snippet[:100])
            if key in seen:
                continue
            seen.add(key)
            merged.append((rel, snippet))
            if len(merged) >= top_k:
                return merged
    return merged


def search_markdown_docs(
    docs_dir: str,
    keyword: str,
    *,
    top_k: int = 3,
    snippet_chars: int = 400,
    max_files: int = 500,
) -> list[tuple[str, str]]:
    """兼容旧调用的便捷入口；生产环境使用 Actor 内驻留的 DocumentSearchIndex。"""
    index = DocumentSearchIndex(docs_dir, max_files=max_files)
    return index.search(keyword, top_k=top_k, snippet_chars=snippet_chars)


def _run_doc_search(
    doc_index: DocumentSearchIndex,
    keyword: str,
    *,
    top_k: int,
    snippet_chars: int,
    split_long_search: bool,
    split_max_len: int,
    allow_fallback: bool,
    fallback_query: str,
) -> tuple[list[tuple[str, str]], str, bool]:
    """执行自主检索；仅当调用方允许且自主检索无结果时额外执行一次 fallback。"""
    queries = (
        _split_search_queries(keyword, max_len=split_max_len)
        if split_long_search and len(keyword.strip()) > split_max_len
        else [keyword]
    )
    merged: list[tuple[str, str]] = []
    for q in queries:
        merged = _merge_search_results(
            merged,
            doc_index.search(q, top_k=top_k, snippet_chars=snippet_chars),
            top_k=top_k,
        )
        if len(merged) >= top_k:
            break

    effective = keyword
    used_fallback = False
    if not merged and allow_fallback:
        fb = fallback_query.strip()
        if fb and _normalize_search_query(fb) != _normalize_search_query(keyword):
            effective = fb
            used_fallback = True
            merged = doc_index.search(fb, top_k=top_k, snippet_chars=snippet_chars)
    return merged, effective, used_fallback


def format_search_results(
    keyword: str,
    results: list[tuple[str, str]],
    *,
    used_query: str = "",
    used_fallback: bool = False,
    fallback_counts_as_hit: bool = False,
    searches_remaining: int = 0,
    result_refs: list[str] | None = None,
    document_candidates: list[tuple[str, str, str]] | None = None,
) -> str:
    if not results:
        if document_candidates:
            lines = [
                "[检索结果]",
                "未找到满足证据门槛的正文片段，但定位到以下候选文档：",
            ]
            for reference, directory, title in document_candidates:
                label = f"{directory} / {title}" if directory else title
                lines.append(f"[{reference}] {label}")
            lines.append(
                "请选择最相关的候选文档执行 read；若方向均不相关则换词 search。"
            )
            return "\n".join(lines)
        lines = [
            "[检索结果]",
            f"未找到与「{keyword}」相关的资料。",
        ]
        if used_fallback:
            lines.append(
                f"三次自主检索均未找到结果，环境已额外使用「{used_query}」兜底检索，仍未命中。"
            )
            lines.append("自主检索额度已耗尽，请根据已有信息输出 \\boxed{答案}。")
        elif searches_remaining > 0:
            lines.append(
                f"还可自主检索 {searches_remaining} 次；请自行判断是换词继续 search，还是根据已有信息作答。"
            )
        else:
            lines.append("自主检索额度已耗尽，请根据已有信息输出 \\boxed{答案}。")
        return "\n".join(lines)
    lines = ["[检索结果]"]
    if used_fallback and used_query:
        lines.append(
            f"（三次自主检索均无结果；环境额外使用 fallback 检索词：「{used_query}」）"
        )
        if not fallback_counts_as_hit:
            lines.append(
                "该结果不计为自主检索命中，且不占三次自主检索额度；请判断片段是否足够并最终作答。"
            )
    # 检索器内部保留来源用于去重和调试，但不向模型暴露路径、文件名或页码。
    bodies = search_result_bodies(results)
    if result_refs:
        for reference, body in zip(result_refs, bodies, strict=False):
            if reference:
                lines.append(f"[{reference}]")
            lines.append(body)
    else:
        lines.extend(bodies)
    if document_candidates:
        lines.append("定位到以下候选文档：")
        for reference, directory, title in document_candidates:
            label = f"{directory} / {title}" if directory else title
            lines.append(f"[{reference}] {label}")
    lines.append(
        "以上为检索返回结果。若片段相关但上下文不足，可 read 对应片段编号；"
        "若需在候选文档内重新定位，可 read 文档编号；方向不相关则换词 search，证据充分则作答；"
        "不要自行续写、补全或复述上述检索内容。"
    )
    return "\n".join(lines)


def _self_test() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        doc_root = Path(tmp)
        (doc_root / "control.md").write_text(
            "控制图分析时，Exclude 功能可以永久排除 Sample。Point Disable 只是临时禁用。",
            encoding="utf-8",
        )
        results = search_markdown_docs(str(doc_root), "Exclude Sample", top_k=2)
        print("search:", results)
        assert results and "control.md" in results[0][0]

        formatted = format_search_results("Exclude Sample", results)
        assert "[检索结果]" in formatted
        assert "control.md" not in formatted
        assert "永久排除 Sample" in formatted
        print("format OK")
        assert extract_boxed("应选 A \\boxed{A}") == "A"
        q = "下面是一道单选题。\n\n题目：Carrier FOUP FOSB 创建\n\n选项：\nA. x"
        assert "Carrier" in _suggest_search_from_query(q)
        assert _is_short_search("ILD", 4)
        assert not _is_short_search("Carrier FOUP", 4)
        assert _split_search_queries("a b c d e f g h i j k l m n o p q r s t", 10)
        assert _is_low_quality_snippet("## Page 1\nSlide number: 3")
        assert not _is_low_quality_snippet("控制图 Exclude 功能可以永久排除 Sample")
        q_opt = "题目：控制图\n\n选项：\nA. Exclude\nB. Point Disable"
        assert _search_biased_by_options("Point Disable", q_opt) == "Point Disable"
        index = DocumentSearchIndex(str(doc_root))
        fb_results, fb_q, used_fallback = _run_doc_search(
            index,
            "不存在的关键词xyz",
            top_k=2,
            snippet_chars=200,
            split_long_search=False,
            split_max_len=30,
            allow_fallback=True,
            fallback_query="Exclude Sample",
        )
        assert fb_results and fb_q == "Exclude Sample" and used_fallback
        q_mrb = "下面是一道填空题。\n\n题目：根据受影响wafer数量，MRB有【1】【2】两种分类"
        mrb_suggest = _suggest_search_from_query(q_mrb)
        assert "【" not in mrb_suggest
        assert "MRB" in mrb_suggest or "wafer" in mrb_suggest
        assert _is_placeholder_search("题干关键词")
        assert not _is_placeholder_search("MRB wafer 数量")
        # 历史轮次里即使出现过 boxed，当前动作仍必须只由最后一条 assistant 消息决定。
        history = [
            {"role": "assistant", "content": r"我先猜 \\boxed{A}"},
            {"role": "environment", "content": "请先检索"},
            {"role": "assistant", "content": "<search>控制图永久排除 Sample</search>"},
        ]
        current = _last_assistant_text(history)
        assert extract_boxed(current) is None
        assert _parse_search_query(current) == "控制图永久排除 Sample"
        nested_example = (
            "我应该使用 <search> 工具，最后执行："
            "<search>刻蚀速率 计算公式</search>"
        )
        assert _parse_search_query(nested_example) == "刻蚀速率 计算公式"
        multiple_actions = "<search>旧词</search> 后改为 <search>新词</search>"
        assert _parse_search_query(multiple_actions) == "新词"
        fire_query = (
            "题目：机台着火了怎么办\n\n选项：\n"
            "A. 使用灭火器灭火\nB. 按emo\nC. 赶紧逃离现场\nD. 通知相关EE"
        )
        assert _answer_evidence_score(
            fire_query,
            "[multiple] B,D",
            "A. 使用灭火器灭火 B. 按emo C. 赶紧逃离现场 D. 通知相关EE",
        ) == 1.0
        assert _answer_evidence_score(
            fire_query,
            "[multiple] B,D",
            "设备机台起火应该先按EMO。",
        ) == 0.5
        assert _answer_evidence_score(
            "题目：SERVER ROOM 通过【1】与Clean room进行连接",
            "[fill] SQL server",
            "SQL server",
        ) == 1.0
        print("qa_agent_env self-test OK")


if __name__ == "__main__":
    _self_test()
    raise SystemExit(0)


import ray
import torch

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn


@ray.remote  # pragma: no cover
class QAAgentEnv(EnvironmentInterface[QAAgentMetadata]):
    """多轮 QA 检索 + 判分环境（Ray Actor）。"""

    def __init__(self, cfg: Optional[dict[str, Any]] = None):
        self.cfg = cfg or {}
        self.docs_dir = str(self.cfg.get("docs_dir", "/data/docs"))
        self.max_searches = int(self.cfg.get("max_searches", 3))
        self.max_reads = int(self.cfg.get("max_reads", 2))
        self.search_document_top_k = int(self.cfg.get("search_document_top_k", 3))
        self.read_context_before = int(self.cfg.get("read_context_before", 2))
        self.read_context_after = int(self.cfg.get("read_context_after", 2))
        self.read_max_chars = int(self.cfg.get("read_max_chars", 1800))
        self.search_top_k = int(self.cfg.get("search_top_k", 3))
        self.snippet_chars = int(self.cfg.get("snippet_chars", 400))
        print(
            "[qa_agent] 开始构建文档索引: "
            f"docs={self.docs_dir} "
            f"max_files={int(self.cfg.get('search_max_files', 5000))} "
            f"max_chunks={int(self.cfg.get('search_max_chunks', 50000))}",
            flush=True,
        )
        self.doc_index = DocumentSearchIndex(
            self.docs_dir,
            max_files=int(self.cfg.get("search_max_files", 5000)),
            max_chunks=int(self.cfg.get("search_max_chunks", 50000)),
            chunk_chars=int(self.cfg.get("search_chunk_chars", 800)),
            overlap_chars=int(self.cfg.get("search_chunk_overlap", 120)),
            per_file_limit=int(self.cfg.get("search_per_file_limit", 2)),
            min_query_coverage=float(self.cfg.get("search_min_query_coverage", 0.35)),
            min_matched_tokens=int(self.cfg.get("search_min_matched_tokens", 2)),
            min_raw_term_coverage=float(
                self.cfg.get("search_min_raw_term_coverage", 0.5)
            ),
            route_directory_count=int(
                self.cfg.get("search_route_directory_count", 3)
            ),
            route_document_count=int(
                self.cfg.get("search_route_document_count", 15)
            ),
            rerank_pool_size=int(self.cfg.get("search_rerank_pool_size", 40)),
            global_supplement_count=int(
                self.cfg.get("search_global_supplement_count", 10)
            ),
            catalog_keyword_count=int(
                self.cfg.get("search_catalog_keyword_count", 12)
            ),
            route_cache_documents=int(
                self.cfg.get("search_route_cache_documents", 32)
            ),
            require_core_term=bool(
                self.cfg.get("search_require_core_term", True)
            ),
            exact_term_boost=float(
                self.cfg.get("search_exact_term_boost", 5.0)
            ),
            global_min_raw_term_coverage=float(
                self.cfg.get("search_global_min_raw_term_coverage", 0.67)
            ),
        )
        self.question_bank_index = QuestionBankIndex(
            self.docs_dir,
            max_files=int(self.cfg.get("search_max_files", 5000)),
        )
        print(
            "[qa_agent] 文档索引: "
            f"files={self.doc_index.files_indexed} chunks={self.doc_index.chunk_count} "
            f"truncated={self.doc_index.truncated}",
            flush=True,
        )
        self.invalid_action_penalty = float(self.cfg.get("invalid_action_penalty", -0.1))
        self.placeholder_query_penalty = float(
            self.cfg.get("placeholder_query_penalty", -0.2)
        )
        self.no_search_before_answer_penalty = float(
            self.cfg.get("no_search_before_answer_penalty", -0.3)
        )
        self.answer_without_evidence_penalty = float(
            self.cfg.get("answer_without_evidence_penalty", -0.2)
        )
        self.mixed_action_penalty = float(
            self.cfg.get("mixed_action_penalty", -0.2)
        )
        self.repeated_query_penalty = float(
            self.cfg.get("repeated_query_penalty", -0.1)
        )
        self.invalid_read_penalty = float(self.cfg.get("invalid_read_penalty", -0.2))
        self.repeated_read_penalty = float(self.cfg.get("repeated_read_penalty", -0.1))
        self.valid_read_bonus = float(self.cfg.get("valid_read_bonus", 0.1))
        self.search_cost = float(self.cfg.get("search_cost", 0.0))
        self.read_cost = float(self.cfg.get("read_cost", 0.0))
        self.fabricated_results_penalty = float(
            self.cfg.get("fabricated_results_penalty", -0.5)
        )
        self.auto_fallback_search = bool(self.cfg.get("auto_fallback_search", True))
        self.fallback_after_failed_searches = int(
            self.cfg.get("fallback_after_failed_searches", self.max_searches)
        )
        self.fallback_penalty = float(self.cfg.get("fallback_penalty", -0.05))
        self.fallback_counts_as_hit = bool(
            self.cfg.get("fallback_counts_as_hit", False)
        )
        self.split_long_search = bool(self.cfg.get("split_long_search", True))
        self.split_max_len = int(self.cfg.get("split_max_len", 30))
        self.use_judge = bool(self.cfg.get("use_judge", True))

        if self.use_judge:
            from common.rewards.qa_judge_reward import qa_judge_reward_fn

            self._reward_fn = qa_judge_reward_fn
        else:
            from common.rewards.qa_reward import qa_rule_reward_fn

            self._reward_fn = qa_rule_reward_fn

    def _score_final(
        self,
        message_log: LLMMessageLogType,
        query: str,
        expected: str,
    ) -> float:
        # 只判最后一轮最终回答，避免历史 search、分析或被拒绝的旧答案污染 reward。
        completion = _last_assistant_text(message_log)
        rewards = self._reward_fn([query], [completion], [expected])
        return float(rewards[0])

    def _step_one(
        self,
        message_log: LLMMessageLogType,
        meta: QAAgentMetadata,
    ) -> tuple[dict[str, str], QAAgentMetadata | None, float, bool]:
        query = str(meta.get("query", ""))
        expected = str(meta.get("expected_answer", ""))
        search_count = int(meta.get("search_count", 0))
        last_search = str(meta.get("last_search_query", ""))
        read_count = int(meta.get("read_count", 0))
        allowed_read_refs = [str(value).upper() for value in meta.get("allowed_read_refs", [])]
        read_history = [str(value).upper() for value in meta.get("read_history", [])]
        read_ref_queries = {str(key).upper(): str(value) for key, value in meta.get("read_ref_queries", {}).items()}
        read_ref_evidence_scores = {
            str(key).upper(): float(value)
            for key, value in meta.get("read_ref_evidence_scores", {}).items()
        }
        read_ref_incomplete = {
            str(key).upper(): bool(value)
            for key, value in meta.get("read_ref_incomplete", {}).items()
        }
        last_search_candidate_only = bool(
            meta.get("last_search_candidate_only", False)
        )
        last_search_read_refs = [
            str(value).upper() for value in meta.get("last_search_read_refs", [])
        ]
        has_answer_evidence = bool(meta.get("has_answer_evidence", False))
        has_search_hit = bool(meta.get("has_search_hit", False))
        last_text = _last_assistant_text(message_log)
        suggest = _suggest_search_from_query(query)

        # 动作解析只看本轮输出。若历史中曾有一个被环境拒绝的 boxed，后续轮次
        # 仍应能执行 search；最终评分时 _score_final 会从完整轨迹取最后一个 boxed。
        boxed = extract_boxed(last_text)
        search_query = _parse_search_query(last_text)
        read_ref = _parse_read_ref(last_text)
        violation = _protocol_violation(
            has_fabricated_results=bool(_FABRICATED_RESULTS_TAG.search(last_text)),
            has_search=search_query is not None,
            has_answer=boxed is not None,
            search_count=search_count,
            search_query=search_query or "",
            last_search_query=last_search,
            has_read=read_ref is not None,
            read_ref=read_ref or "",
            allowed_read_refs=allowed_read_refs,
            read_history=read_history,
        )
        if violation is not None:
            violation_responses = {
                PROTOCOL_FABRICATED_RESULTS: (
                    "禁止由 Assistant 自行输出环境结果的边界标记或伪造环境内容。"
                    "请重新输出合法的 search、read 或最终 boxed 动作。",
                    self.fabricated_results_penalty,
                ),
                PROTOCOL_MIXED_ACTION: (
                    "同一轮只能执行一种动作，不得混合 search、read 和 boxed。"
                    "请重新选择一个合法动作。",
                    self.mixed_action_penalty,
                ),
                PROTOCOL_ANSWER_BEFORE_SEARCH: (
                    "必须至少完成一次真实检索后才能提交最终答案。"
                    "请先输出 <search>关键词</search>。",
                    self.no_search_before_answer_penalty,
                ),
                PROTOCOL_REPEATED_QUERY: (
                    f"您已搜过「{search_query}」。请更换检索词，"
                    "或根据已有结果输出最终 \\boxed{答案}。",
                    self.repeated_query_penalty,
                ),
                PROTOCOL_READ_BEFORE_SEARCH: ("必须先用 search 获得结果编号才能 read。", self.invalid_read_penalty),
                PROTOCOL_UNKNOWN_READ: ("该编号不是本题此前 search 返回的结果。", self.invalid_read_penalty),
                PROTOCOL_REPEATED_READ: (f"您已读取过 {read_ref}，请勿重复 read。", self.repeated_read_penalty),
            }
            content, penalty = violation_responses[violation]
            return (
                {"role": "environment", "content": content},
                meta,
                penalty,
                False,
            )

        if boxed is not None:
            base_reward = self._score_final(message_log, query, expected)
            reward = base_reward
            has_usable_evidence = has_search_hit or read_count > 0
            answered_before_evidence_exhausted = (
                search_count > 0
                and not has_usable_evidence
                and int(meta.get("fallback_count", 0)) == 0
                and search_count < self.fallback_after_failed_searches
            )
            if answered_before_evidence_exhausted:
                reward += self.answer_without_evidence_penalty
            # 暂停延迟发放 valid_read_bonus；保留判定状态，后续取消注释即可恢复。
            # if base_reward >= 1.0 and bool(meta.get("pending_valid_read_bonus", False)):
            #     reward += self.valid_read_bonus
            obs = {"role": "environment", "content": f"得分: {reward:.3f}"}
            return obs, None, reward, True

        if read_ref is not None:
            if read_count >= self.max_reads:
                return ({"role": "environment", "content": f"已达最大阅读次数（{self.max_reads}），请继续 search 或作答。"}, meta, 0.0, False)
            resolved_ref = read_ref
            if ":" in read_ref:
                context = self.doc_index.read_context(read_ref, before=self.read_context_before, after=self.read_context_after, max_chars=self.read_max_chars)
            else:
                located = self.doc_index.read_document_context(
                    read_ref,
                    read_ref_queries.get(read_ref, last_search),
                    before=self.read_context_before,
                    after=self.read_context_after,
                    max_chars=self.read_max_chars,
                )
                if located:
                    resolved_ref, context = located
                else:
                    context = None
            if not context:
                return ({"role": "environment", "content": f"无法读取 {read_ref}，请使用其他已返回编号或重新 search。"}, meta, self.invalid_read_penalty, False)
            new_meta = {**meta, "read_count": read_count + 1, "read_history": [*read_history, read_ref]}
            content = f"[阅读内容 {resolved_ref}]\n{context}\n以上为命中位置附近的正文；证据充分则作答，不充分则选择其他编号 read 或换词 search。"
            # 暂停 read 证据增量检测及延迟奖励状态记录；保留代码便于后续恢复。
            # baseline_score = read_ref_evidence_scores.get(read_ref, 0.0)
            # read_score = _answer_evidence_score(query, expected, context)
            # qualifies_for_delayed_bonus = (
            #     read_ref in last_search_read_refs
            #     and read_ref_incomplete.get(read_ref, False)
            #     and read_score > baseline_score + 1e-9
            #     and read_score > 0.0
            # )
            # if qualifies_for_delayed_bonus:
            #     new_meta["pending_valid_read_bonus"] = True
            # if read_score > 0.0:
            #     new_meta["has_answer_evidence"] = True
            return (
                {"role": "environment", "content": content},
                new_meta,
                self.read_cost,
                False,
            )

        if search_query is not None:
            if _is_literal_search_placeholder(search_query):
                return (
                    {
                        "role": "environment",
                        "content": (
                            "“检索词”是动作格式中的占位文字，不能作为实际 query。"
                            "请根据题目对象和所问属性给出具体检索词。"
                        ),
                    },
                    meta,
                    self.placeholder_query_penalty,
                    False,
                )
            if search_count >= self.max_searches:
                content = (
                    f"[检索结果]\n"
                    f"已达最大检索次数（{self.max_searches}），不得继续 search。"
                    f"请根据已有信息直接输出 \\boxed{{答案}}。"
                )
                return (
                    {"role": "environment", "content": content},
                    meta,
                    0.0,  # 暂不惩罚超过最大搜索次数；仍拦截执行并提示最终作答。
                    False,
                )

            # 除字面占位符“检索词”外，其余 query 质量由 Agent 自主负责，
            # 直接交给检索器执行；System Prompt 仍保留软约束。

            next_search_count = search_count + 1
            allow_fallback = _fallback_eligible(
                enabled=self.auto_fallback_search,
                has_prior_search_hit=has_search_hit,
                search_count_after_current=next_search_count,
                fallback_after_failed_searches=self.fallback_after_failed_searches,
            )
            results, effective_query, used_fallback = _run_doc_search(
                self.doc_index,
                search_query,
                top_k=self.search_top_k,
                snippet_chars=self.snippet_chars,
                split_long_search=self.split_long_search,
                split_max_len=self.split_max_len,
                allow_fallback=allow_fallback,
                fallback_query=suggest,
            )
            # Answer-key files often contain only compact answer sequences.  The
            # paired question-bank index reconstructs question + answer records
            # before retrieval, so a hit on a paper can expose its actual answer.
            if not used_fallback:
                paired_results = self.question_bank_index.search(
                    search_query,
                    top_k=int(self.cfg.get("question_bank_top_k", 2)),
                )
                results = _merge_search_results(
                    paired_results,
                    results,
                    top_k=self.search_top_k,
                )
            result_refs = [
                self.doc_index.reference_for_result(source, snippet) or ""
                for source, snippet in results
            ]
            document_candidates = self.doc_index.document_candidates(
                effective_query, top_k=self.search_document_top_k
            )
            content = format_search_results(
                search_query,
                results,
                used_query=effective_query,
                used_fallback=used_fallback,
                fallback_counts_as_hit=self.fallback_counts_as_hit,
                searches_remaining=max(0, self.max_searches - next_search_count),
                result_refs=result_refs,
                document_candidates=document_candidates,
            )
            credited_hit = _credited_search_hit(
                results_found=bool(results),
                used_fallback=used_fallback,
                fallback_counts_as_hit=self.fallback_counts_as_hit,
            )
            result_evidence_scores = {
                ref: _answer_evidence_score(query, expected, snippet)
                for ref, (_, snippet) in zip(result_refs, results, strict=False)
                if ref
            }
            candidate_refs = [
                reference for reference, _, _ in document_candidates
            ]
            candidate_only = not results and bool(candidate_refs)
            current_read_refs = [
                *(ref for ref in result_refs if ref),
                *candidate_refs,
            ]
            new_meta: QAAgentMetadata = {
                **meta,
                "search_count": next_search_count,
                "last_search_query": search_query,
                "has_search_hit": has_search_hit or credited_hit,
                "has_answer_evidence": has_answer_evidence
                or any(score > 0.0 for score in result_evidence_scores.values()),
                "fallback_count": int(meta.get("fallback_count", 0))
                + int(used_fallback),
                "allowed_read_refs": list(dict.fromkeys([
                    *allowed_read_refs,
                    *(ref for ref in result_refs if ref),
                    *(reference for reference, _, _ in document_candidates),
                ])),
                "read_ref_queries": {
                    **read_ref_queries,
                    **{reference: effective_query for reference, _, _ in document_candidates},
                },
                "read_ref_evidence_scores": {
                    **read_ref_evidence_scores,
                    **result_evidence_scores,
                    **{reference: 0.0 for reference in candidate_refs},
                },
                "read_ref_incomplete": {
                    **read_ref_incomplete,
                    **{
                        reference: score < 1.0
                        for reference, score in result_evidence_scores.items()
                    },
                    **{
                        reference: candidate_only
                        for reference in candidate_refs
                    },
                },
                "last_search_candidate_only": candidate_only,
                "last_search_read_refs": list(dict.fromkeys(current_read_refs)),
            }
            search_reward = self.search_cost + (
                self.fallback_penalty if used_fallback else 0.0
            )
            return (
                {"role": "environment", "content": content},
                new_meta,
                search_reward,
                False,
            )

        if search_count > 0:
            content = (
                "请自行判断环境返回的证据是否充分：若片段相关但上下文不足，可输出 <read>结果编号</read>；"
                "输出新的 <search>关键词</search>；若充分，输出最终 \\boxed{答案}。"
            )
        else:
            content = (
                "请先从题目提取专业名词检索，例如 "
                f"<search>{suggest}</search>。"
                "不要把「题干关键词」等说明文字原样当作 search 内容。"
            )
        return (
            {"role": "environment", "content": content},
            meta,
            self.invalid_action_penalty,
            False,
        )

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[QAAgentMetadata],
    ) -> EnvironmentReturn[QAAgentMetadata]:
        observations: list[dict[str, str]] = []
        next_metadata: list[QAAgentMetadata | None] = []
        rewards: list[float] = []
        terminateds: list[bool] = []
        expected_answers: list[str] = []

        for log, meta in zip(message_log_batch, metadata, strict=False):
            meta = dict(meta or {})
            meta.setdefault("search_count", 0)
            meta.setdefault("last_search_query", "")
            meta.setdefault("has_search_hit", False)
            meta.setdefault("fallback_count", 0)
            meta.setdefault("read_count", 0)
            meta.setdefault("allowed_read_refs", [])
            meta.setdefault("read_history", [])
            meta.setdefault("read_ref_queries", {})
            meta.setdefault("read_ref_evidence_scores", {})
            meta.setdefault("read_ref_incomplete", {})
            meta.setdefault("last_search_candidate_only", False)
            meta.setdefault("last_search_read_refs", [])
            meta.setdefault("has_answer_evidence", False)
            # meta.setdefault("pending_valid_read_bonus", False)
            obs, new_meta, reward, terminated = self._step_one(log, meta)
            observations.append(obs)
            next_metadata.append(new_meta)
            rewards.append(reward)
            terminateds.append(terminated)
            expected_answers.append(str(meta.get("expected_answer", "")))

        return EnvironmentReturn(
            observations=observations,
            metadata=next_metadata,
            # NeMo-RL 会用该字段覆盖下一轮 stop_strings；None 表示取消而非沿用。
            # 因此所有仍在继续的 episode 都必须显式保留 </search>。
            next_stop_strings=_next_action_stop_strings(terminateds),
            rewards=torch.tensor(rewards, dtype=torch.float32),
            terminateds=torch.tensor(terminateds, dtype=torch.bool),
            answers=expected_answers,
        )

    def shutdown(self):
        pass

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> tuple[BatchedDataDict, dict]:
        rewards = batch.get(
            "total_reward", torch.tensor([0.0] * len(batch["idx"]))
        ).float()
        if len(rewards) == 0:
            return batch, {}
        metrics = {
            "qa_agent_mean_reward": rewards.mean().item(),
            "qa_agent_perfect_rate": (rewards >= 1.0).float().mean().item(),
            "qa_agent_format_penalty_rate": (rewards < 0).float().mean().item(),
        }
        return batch, metrics
