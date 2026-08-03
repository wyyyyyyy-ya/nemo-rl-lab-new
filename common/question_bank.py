"""Pair question-paper and answer-key files into searchable Q&A records.

The pairing/parsing strategy is adapted from lihaozhang01/nemo-rl-lab-exam
commit 9931f2a2a0d93723f8771796cc49e0d5711f900f.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from common.doc_search import normalize_text, tokenize

_QUESTION_FILE_RE = re.compile(r"试题|试卷|考核|考试", re.I)
_ANSWER_FILE_RE = re.compile(r"答案|参考答案|答案版", re.I)
_PAIR_WORD_RE = re.compile(r"参考答案|答案版|答案|试题版|试题|试卷|考核试卷|考试", re.I)
_TYPE_RE = re.compile(r"(选择题|单选题|多选题|填空题|判断题|简答题)", re.I)
_QUESTION_RE = re.compile(r"^\s*(\d+)[.、．]\s*(.+)")
_OPTION_RE = re.compile(r"^\s*([A-Ha-h])[.、．]\s*(.+)")


def _pair_key(path: Path) -> str:
    stem = _PAIR_WORD_RE.sub("", path.stem)
    return re.sub(r"[\s_\-—（）()\[\]]+", "", stem).casefold()


def _kind(text: str) -> str:
    if "填空" in text:
        return "填空题"
    if "判断" in text:
        return "判断题"
    if "简答" in text:
        return "简答题"
    return "选择题"


@dataclass(frozen=True)
class QuestionAnswerItem:
    question: str
    answer: str
    question_path: str
    answer_path: str
    options: tuple[str, ...] = ()

    @property
    def text(self) -> str:
        lines = [f"题目：{self.question}"]
        lines.extend(self.options)
        lines.append(f"答案：{self.answer}")
        return "\n".join(lines)


def _parse_questions(text: str, path: Path) -> list[tuple[str, str, tuple[str, ...]]]:
    lines = [line.strip() for line in text.replace("\\n", "\n").splitlines()]
    items: list[tuple[str, str, tuple[str, ...]]] = []
    current_kind = "选择题"
    current_question = ""
    current_options: list[str] = []

    def flush() -> None:
        nonlocal current_question, current_options
        if current_question:
            items.append((current_kind, current_question, tuple(current_options)))
        current_question, current_options = "", []

    for line in lines:
        if not line:
            continue
        type_match = _TYPE_RE.search(line)
        # A section heading is short and does not itself contain a real question.
        if type_match and len(line) <= 24 and not re.search(r"[？?（(]", line):
            flush()
            current_kind = _kind(type_match.group(1))
            continue
        option = _OPTION_RE.match(line)
        if option and current_question:
            parts = re.split(r"(?=\b[a-hA-H][.、．])", line)
            current_options.extend(part.strip() for part in parts if part.strip())
            continue
        question = _QUESTION_RE.match(line)
        if question:
            body = question.group(2).strip()
            # Numbered section headers such as "1. 选择题".
            header = _TYPE_RE.match(body)
            if header and body.startswith(header.group(1)):
                flush()
                current_kind = _kind(header.group(1))
                continue
            flush()
            current_question = body
            continue
        # OCR often puts all options on the same line.
        if current_question and re.search(r"\b[a-hA-H][.、．]", line):
            parts = re.split(r"(?=\b[a-hA-H][.、．])", line)
            current_options.extend(part.strip() for part in parts if part.strip())
    flush()
    return items


def _split_choice_answers(text: str) -> list[str]:
    compact = re.sub(r"\s+", "", text).upper()
    return list(re.sub(r"[^A-H]", "", compact))


def _parse_answers(text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current = ""
    buffer: list[str] = []

    def flush() -> None:
        if not current:
            return
        if current == "选择题":
            sections[current] = _split_choice_answers("\n".join(buffer))
        elif current == "判断题":
            compact = "".join(buffer)
            sections[current] = [
                "正确" if char in "√✓" else "错误"
                for char in compact
                if char in "√✓×Xx"
            ]
        else:
            sections[current] = [
                re.sub(r"^\s*\d+[.、．]\s*", "", line).strip()
                for line in buffer
                if line.strip()
            ]

    for raw_line in text.replace("\\n", "\n").splitlines():
        line = raw_line.strip().strip("*")
        match = _TYPE_RE.search(line)
        if match and len(line) <= 24:
            flush()
            current, buffer = _kind(match.group(1)), []
        elif current and line:
            buffer.append(line)
    flush()
    return sections


class QuestionBankIndex:
    """Small in-memory index of Q&A records reconstructed from paired files."""

    def __init__(self, docs_dir: str, *, max_files: int = 5000) -> None:
        self.root = Path(docs_dir)
        self.items: list[QuestionAnswerItem] = []
        self.pairs: list[tuple[Path, Path]] = []
        if not self.root.is_dir():
            return
        paths = sorted(
            (
                path
                for pattern in ("*.md", "*.txt")
                for path in self.root.rglob(pattern)
            ),
            key=lambda path: str(path).casefold(),
        )[:max_files]
        question_files = [p for p in paths if _QUESTION_FILE_RE.search(p.stem) and not _ANSWER_FILE_RE.search(p.stem)]
        answer_files = [p for p in paths if _ANSWER_FILE_RE.search(p.stem)]
        for answer_path in answer_files:
            candidates = [p for p in question_files if p.parent == answer_path.parent] or question_files
            ranked = sorted(
                (
                    (SequenceMatcher(None, _pair_key(answer_path), _pair_key(p)).ratio(), p)
                    for p in candidates
                ),
                key=lambda row: (-row[0], str(row[1]).casefold()),
            )
            if not ranked:
                continue
            best_score, question_path = ranked[0]
            exact = _pair_key(answer_path) == _pair_key(question_path)
            margin = best_score - (ranked[1][0] if len(ranked) > 1 else 0.0)
            if not exact and (best_score < 0.72 or margin < 0.08):
                continue
            self.pairs.append((question_path, answer_path))
            self._add_pair(question_path, answer_path)

    def _add_pair(self, question_path: Path, answer_path: Path) -> None:
        questions = _parse_questions(question_path.read_text(encoding="utf-8", errors="ignore"), question_path)
        answers = _parse_answers(answer_path.read_text(encoding="utf-8", errors="ignore"))
        positions: dict[str, int] = {}
        for kind, question, options in questions:
            position = positions.get(kind, 0)
            positions[kind] = position + 1
            kind_answers = answers.get(kind, [])
            if position >= len(kind_answers):
                continue
            answer = kind_answers[position]
            if answer:
                self.items.append(
                    QuestionAnswerItem(
                        question=question,
                        answer=answer,
                        question_path=str(question_path),
                        answer_path=str(answer_path),
                        options=options,
                    )
                )

    def search(self, query: str, *, top_k: int = 2) -> list[tuple[str, str]]:
        query = query.strip()
        query_tokens = set(tokenize(query))
        query_norm = normalize_text(query)
        if not query_tokens:
            return []
        ranked: list[tuple[float, int, QuestionAnswerItem]] = []
        for ordinal, item in enumerate(self.items):
            question_tokens = set(tokenize(item.question))
            coverage = len(query_tokens & question_tokens) / len(query_tokens)
            similarity = SequenceMatcher(None, query_norm, normalize_text(item.question)).ratio()
            phrase = 1.0 if query_norm and query_norm in normalize_text(item.question) else 0.0
            score = 5.0 * coverage + 3.0 * similarity + 2.0 * phrase
            if coverage < 0.45 and similarity < 0.62:
                continue
            ranked.append((-score, ordinal, item))
        ranked.sort()
        return [
            (f"{item.question_path} | answer={item.answer_path}", item.text)
            for _, _, item in ranked[: max(1, top_k)]
        ]
