"""分层文档检索：文档路由、候选段落召回、全库补充和证据重排。"""

from __future__ import annotations

import math
import re
from collections import Counter, OrderedDict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path, PurePosixPath

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9._+-]*|[\u4e00-\u9fff]+", re.I)
_SPACE_RE = re.compile(r"\s+")
_PAGE_RE = re.compile(
    r"^\s*(?:<!--\s*)?##?\s*Page\s+(\d+)\s*(?:-->)?\s*$",
    re.I,
)
_SLIDE_RE = re.compile(
    r"^\s*(?:<!--\s*)?Slide number:\s*(\d+)\s*(?:-->)?\s*$",
    re.I,
)
_SLIDE_TITLE_RE = re.compile(r"^\s*(?:Slide title|幻灯片标题)\s*[:：]\s*(.+)$", re.I)
_MARKDOWN_TITLE_RE = re.compile(r"^\s*#{1,6}\s+(.+?)\s*$")
_DROP_LINE_RE = re.compile(
    r"^\s*\*?(?:\[(?:End|Image) OCR\]|目录)\*?\s*$",
    re.I,
)
_RAW_TERM_SPLIT_RE = re.compile(r"[\s,，、/；;:：()（）]+")
_CATALOG_STOP_TOKENS = {
    "一个",
    "以及",
    "进行",
    "使用",
    "可以",
    "相关",
    "内容",
    "介绍",
    "说明",
    "要求",
    "管理",
}
_GENERIC_HEADINGS = {"notes", "备注"}
_QUERY_ATTRIBUTE_TERMS = {
    "作用",
    "功能",
    "定义",
    "含义",
    "意思",
    "要求",
    "规定",
    "分类",
    "类型",
    "数量",
    "比例",
    "原因",
    "目的",
    "方法",
    "方式",
    "创建",
    "创建方式",
    "源文件",
    "步骤",
    "流程",
    "区别",
    "关系",
    "是否",
    "哪个",
    "哪些",
    "什么",
    "多少",
    "如何",
    "怎么",
    "why",
    "what",
    "which",
    "how",
    "function",
    "purpose",
    "definition",
    "meaning",
    "quantity",
    "ratio",
    "type",
    "types",
}
_QUERY_EXPANSIONS = {
    "作用": ("功能", "原理", "function", "purpose"),
    "分类": ("类型", "类别", "category", "classification", "type"),
    "数量": ("数目", "count", "number", "quantity"),
    "比例": ("占比", "百分比", "percentage", "ratio"),
    "创建": ("create", "creation"),
    "源文件": ("source file", "original file"),
}
_DOT_LEADER_RE = re.compile(r"(?:\.{5,}|…{3,})\s*\d+")
_SENTENCE_END_RE = re.compile(r"[。！？；.!?;]")


def normalize_text(text: str) -> str:
    """保留中英文与数字，去掉空白/标点，用于完整术语命中。"""
    return "".join(_TOKEN_RE.findall(text.lower()))


def tokenize(text: str) -> list[str]:
    """英文按词、中文按二元字切分，无需额外分词依赖。"""
    tokens: list[str] = []
    for part in _TOKEN_RE.findall(text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", part):
            if len(part) == 1:
                tokens.append(part)
            else:
                tokens.extend(part[i : i + 2] for i in range(len(part) - 1))
                if len(part) <= 6:
                    tokens.append(part)
        else:
            tokens.append(part)
    return tokens


def is_low_quality_snippet(snippet: str) -> bool:
    text = snippet.strip()
    if len(text) < 24:
        return True
    visible_lines = [
        line
        for line in text.splitlines()
        if not (_DROP_LINE_RE.match(line) or _PAGE_RE.match(line) or _SLIDE_RE.match(line))
    ]
    visible = " ".join(visible_lines).strip()
    if len(visible) < 24 or visible.count("…") > 4:
        return True
    # 目录页通常包含多组点线+页码；它可用于文档路由，但不应作为最终证据。
    if len(_DOT_LEADER_RE.findall(visible)) >= 2:
        return True
    punctuation = visible.count(".") + visible.count("…")
    if punctuation / max(1, len(visible)) > 0.18:
        return True
    lexical_tokens = _TOKEN_RE.findall(visible.lower())
    if lexical_tokens:
        most_common_count = Counter(lexical_tokens).most_common(1)[0][1]
        if most_common_count >= 6 and most_common_count / len(lexical_tokens) > 0.2:
            return True
    return False


def search_result_bodies(results: list[tuple[str, str]]) -> list[str]:
    """只返回证据正文，来源信息仅供检索器内部去重和调试。"""
    return [snippet for _, snippet in results]


def _split_long_text(text: str, chunk_chars: int, overlap_chars: int) -> list[str]:
    if len(text) <= chunk_chars:
        return [text]
    step = max(1, chunk_chars - overlap_chars)
    return [text[start : start + chunk_chars] for start in range(0, len(text), step)]


def _document_blocks(text: str) -> list[tuple[str, str]]:
    """返回清洗后的语义块及其 Page/Slide 位置。"""
    # 部分 PPT 转换文档把换行保存成字面量 ``\n``。仅在它明显占主导时
    # 还原，避免误改代码类文档中偶尔出现的转义字符串。
    escaped_newlines = text.count("\\n")
    if escaped_newlines >= 3:
        text = text.replace("\\r\\n", "\n").replace("\\n", "\n")
    blocks: list[tuple[str, str]] = []
    current_lines: list[str] = []
    location = ""

    def flush() -> None:
        if current_lines:
            block = "\n".join(current_lines).strip()
            if block:
                blocks.append((block, location))
            current_lines.clear()

    for line in text.splitlines():
        page = _PAGE_RE.match(line)
        slide = _SLIDE_RE.match(line)
        if page or slide:
            flush()
            location = f"Page {page.group(1)}" if page else f"Slide {slide.group(1)}"
            continue
        if _DROP_LINE_RE.match(line):
            continue
        if not line.strip():
            flush()
            continue
        current_lines.append(line)
    flush()
    return blocks


def _split_document_with_locations(
    text: str, *, chunk_chars: int, overlap_chars: int
) -> list[tuple[str, str]]:
    chunks: list[tuple[str, str]] = []
    heading = ""
    for paragraph, location in _document_blocks(text):
        markdown_title = _MARKDOWN_TITLE_RE.match(paragraph)
        slide_title = _SLIDE_TITLE_RE.match(paragraph)
        if (markdown_title or slide_title) and len(paragraph) <= 160:
            title_text = (
                slide_title.group(1) if slide_title else markdown_title.group(1)
            ).strip(" :：").lower()
            heading = "" if title_text in _GENERIC_HEADINGS else paragraph
            continue
        section = f"{heading}\n\n{paragraph}" if heading else paragraph
        for chunk in _split_long_text(section, chunk_chars, overlap_chars):
            if not is_low_quality_snippet(chunk):
                chunks.append((chunk, location))
    return chunks


def split_document(text: str, *, chunk_chars: int, overlap_chars: int) -> list[str]:
    """按语义段落组块；标题附到下一段，超长段落使用带重叠滑窗。"""
    return [
        chunk
        for chunk, _ in _split_document_with_locations(
            text,
            chunk_chars=chunk_chars,
            overlap_chars=overlap_chars,
        )
    ]


def _first_effective_title(text: str) -> str:
    for line in text.splitlines():
        if _PAGE_RE.match(line) or _SLIDE_RE.match(line) or _DROP_LINE_RE.match(line):
            continue
        slide_title = _SLIDE_TITLE_RE.match(line)
        markdown_title = _MARKDOWN_TITLE_RE.match(line)
        if slide_title:
            return slide_title.group(1).strip()
        if markdown_title:
            return markdown_title.group(1).strip()
    return ""


def _document_titles(text: str, limit: int = 24) -> list[str]:
    titles: list[str] = []
    for line in text.splitlines():
        slide_title = _SLIDE_TITLE_RE.match(line)
        markdown_title = _MARKDOWN_TITLE_RE.match(line)
        title = ""
        if slide_title:
            title = slide_title.group(1).strip()
        elif markdown_title and not _PAGE_RE.match(line):
            title = markdown_title.group(1).strip()
        if title.strip(" :：").lower() in _GENERIC_HEADINGS:
            continue
        if title and title not in titles:
            titles.append(title)
        if len(titles) >= limit:
            break
    return titles


def _catalog_keywords(text: str, limit: int = 12) -> list[str]:
    return _catalog_keywords_from_tokens(tokenize(text), limit)


def _catalog_keywords_from_tokens(tokens: list[str], limit: int = 12) -> list[str]:
    counts = Counter(
        token
        for token in tokens
        if len(token) >= 2 and token not in _CATALOG_STOP_TOKENS
    )
    return [token for token, _ in counts.most_common(limit)]


def _query_terms(query: str) -> tuple[set[str], set[str]]:
    """返回原始短语与核心实体短语；属性/疑问词不能单独证明相关。"""
    raw_terms = {
        term.strip().lower()
        for term in _RAW_TERM_SPLIT_RE.split(query)
        if len(normalize_text(term)) >= 2
    }
    core_terms = {
        term
        for term in raw_terms
        if normalize_text(term) not in _QUERY_ATTRIBUTE_TERMS
    }
    return raw_terms, core_terms


def _expanded_query_tokens(query: str) -> set[str]:
    tokens = set(tokenize(query))
    lowered = query.lower()
    for needle, expansions in _QUERY_EXPANSIONS.items():
        if needle in lowered:
            for expansion in expansions:
                tokens.update(tokenize(expansion))
    return tokens


def _term_coverage(terms: set[str], normalized_text: str) -> float:
    if not terms:
        return 0.0
    return sum(normalize_text(term) in normalized_text for term in terms) / len(terms)


def _near_duplicate(left: str, right: str) -> bool:
    """最终少量证据做近似去重，避免重复课件或重叠 chunk 占满 Top-K。"""
    left_norm = normalize_text(left)
    right_norm = normalize_text(right)
    if not left_norm or not right_norm:
        return False
    shorter, longer = sorted((left_norm, right_norm), key=len)
    if len(shorter) >= 48 and shorter in longer:
        return True
    return SequenceMatcher(None, left_norm, right_norm).ratio() >= 0.82


@dataclass(frozen=True)
class _IndexedDocument:
    path: str
    directory: str
    title: str
    catalog_text: str
    normalized: str
    term_counts: Counter[str]
    body_token_hashes: frozenset[int]
    length: int


@dataclass(frozen=True)
class _IndexedChunk:
    document_id: int
    path: str
    location: str
    text: str
    normalized: str
    term_counts: Counter[str]
    length: int


def _evenly_spaced_chunks(
    chunks: list[tuple[str, str]], limit: int
) -> list[tuple[str, str]]:
    """在整篇文档中均匀取样，避免轻量全库索引只保留文档开头。"""
    if limit <= 0:
        return []
    if len(chunks) <= limit:
        return chunks
    if limit == 1:
        return [chunks[len(chunks) // 2]]
    indexes = {
        round(position * (len(chunks) - 1) / (limit - 1))
        for position in range(limit)
    }
    return [chunks[index] for index in sorted(indexes)]


class DocumentSearchIndex:
    """驻留内存的分层 BM25 索引；search 阶段不再访问磁盘。"""

    def __init__(
        self,
        docs_dir: str,
        *,
        max_files: int = 5000,
        max_chunks: int = 50000,
        chunk_chars: int = 800,
        overlap_chars: int = 120,
        per_file_limit: int = 2,
        min_query_coverage: float = 0.35,
        min_matched_tokens: int = 2,
        min_raw_term_coverage: float = 0.5,
        route_directory_count: int = 3,
        route_document_count: int = 15,
        rerank_pool_size: int = 40,
        global_supplement_count: int = 10,
        catalog_keyword_count: int = 12,
        route_cache_documents: int = 32,
        require_core_term: bool = True,
        exact_term_boost: float = 5.0,
        global_min_raw_term_coverage: float = 0.67,
    ) -> None:
        self.root = Path(docs_dir)
        self.max_files = max(1, max_files)
        self.max_chunks = max(1, max_chunks)
        self.chunk_chars = max(100, chunk_chars)
        self.overlap_chars = max(0, min(overlap_chars, self.chunk_chars // 2))
        self.per_file_limit = max(1, per_file_limit)
        self.min_query_coverage = max(0.0, min(1.0, min_query_coverage))
        self.min_matched_tokens = max(1, min_matched_tokens)
        self.min_raw_term_coverage = max(0.0, min(1.0, min_raw_term_coverage))
        self.route_directory_count = max(1, route_directory_count)
        self.route_document_count = max(1, route_document_count)
        self.rerank_pool_size = max(1, rerank_pool_size)
        self.global_supplement_count = max(1, global_supplement_count)
        self.catalog_keyword_count = max(0, catalog_keyword_count)
        self.route_cache_documents = max(1, route_cache_documents)
        self.require_core_term = bool(require_core_term)
        self.exact_term_boost = max(0.0, exact_term_boost)
        self.global_min_raw_term_coverage = max(
            0.0, min(1.0, global_min_raw_term_coverage)
        )
        self.files_indexed = 0
        self.truncated = False
        self._documents: list[_IndexedDocument] = []
        self._chunks: list[_IndexedChunk] = []
        self._doc_catalog_freq: Counter[str] = Counter()
        self._chunk_freq: Counter[str] = Counter()
        self._avg_catalog_length = 1.0
        self._avg_chunk_length = 1.0
        self._route_chunk_cache: OrderedDict[int, list[_IndexedChunk]] = OrderedDict()
        self._build()

    @property
    def document_count(self) -> int:
        return len(self._documents)

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def _build(self) -> None:
        if not self.root.is_dir():
            return
        paths = sorted(
            {
                path
                for pattern in ("*.md", "*.txt")
                for path in self.root.rglob(pattern)
            },
            key=lambda path: str(path.relative_to(self.root)).lower(),
        )
        if len(paths) > self.max_files:
            paths = paths[: self.max_files]
            self.truncated = True

        # 均匀分配 chunk 预算，避免按路径排序靠前的文档吃完整个额度。
        chunks_per_document = max(1, self.max_chunks // max(1, len(paths)))
        total_catalog_length = 0
        total_chunk_length = 0
        for path in paths:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

            rel = path.relative_to(self.root).as_posix()
            pure_path = PurePosixPath(rel)
            directory = pure_path.parent.as_posix()
            if directory == ".":
                directory = ""
            titles = _document_titles(text)
            title = titles[0] if titles else _first_effective_title(text)
            body_tokens = tokenize(text)
            keywords = _catalog_keywords_from_tokens(
                body_tokens, self.catalog_keyword_count
            )
            catalog_text = " ".join(
                part
                for part in (
                    rel,
                    " ".join(pure_path.parts[:-1]),
                    pure_path.stem,
                    title,
                    " ".join(titles),
                    " ".join(keywords),
                )
                if part
            )
            catalog_counts = Counter(tokenize(catalog_text))
            document_id = len(self._documents)
            document = _IndexedDocument(
                path=rel,
                directory=directory,
                title=title,
                catalog_text=catalog_text,
                normalized=normalize_text(catalog_text),
                term_counts=catalog_counts,
                # 轻量全文词汇签名只用于第一级路由。它让深页中的专业词也能
                # 把文档送入第二级全文 chunk 搜索，而无需常驻全部正文。
                body_token_hashes=frozenset(hash(token) for token in body_tokens),
                length=max(1, sum(catalog_counts.values())),
            )
            self._documents.append(document)
            self._doc_catalog_freq.update(catalog_counts.keys())
            total_catalog_length += document.length
            self.files_indexed += 1

            available = max(0, self.max_chunks - len(self._chunks))
            if available <= 0:
                self.truncated = True
                continue
            document_chunks = _split_document_with_locations(
                text,
                chunk_chars=self.chunk_chars,
                overlap_chars=self.overlap_chars,
            )
            selected = _evenly_spaced_chunks(
                document_chunks,
                min(chunks_per_document, available),
            )
            if len(selected) < len(document_chunks):
                self.truncated = True
            for chunk_text, location in selected:
                counts = Counter(tokenize(chunk_text))
                if not counts:
                    continue
                chunk = _IndexedChunk(
                    document_id=document_id,
                    path=rel,
                    location=location,
                    text=chunk_text,
                    normalized=normalize_text(chunk_text),
                    term_counts=counts,
                    length=sum(counts.values()),
                )
                self._chunks.append(chunk)
                self._chunk_freq.update(counts.keys())
                total_chunk_length += chunk.length

        if self._documents:
            self._avg_catalog_length = total_catalog_length / len(self._documents)
        if self._chunks:
            self._avg_chunk_length = total_chunk_length / len(self._chunks)

    @staticmethod
    def _bm25(
        term_counts: Counter[str],
        length: int,
        query_tokens: set[str],
        doc_freq: Counter[str],
        corpus_size: int,
        average_length: float,
    ) -> tuple[float, int]:
        score = 0.0
        matched = 0
        k1, b = 1.5, 0.75
        for token in query_tokens:
            tf = term_counts.get(token, 0)
            if not tf:
                continue
            matched += 1
            df = doc_freq[token]
            idf = math.log(1.0 + (corpus_size - df + 0.5) / (df + 0.5))
            denom = tf + k1 * (1.0 - b + b * length / average_length)
            score += idf * tf * (k1 + 1.0) / denom
        return score, matched

    def _route_documents(
        self,
        query: str,
        query_tokens: set[str],
        query_norm: str,
        raw_terms: set[str],
        core_terms: set[str],
    ) -> tuple[set[int], dict[int, float]]:
        document_scores: dict[int, float] = {}
        ranked_documents: list[tuple[float, str, int]] = []
        directory_scores: dict[str, float] = {}
        for document_id, document in enumerate(self._documents):
            bm25, matched = self._bm25(
                document.term_counts,
                document.length,
                query_tokens,
                self._doc_catalog_freq,
                len(self._documents),
                self._avg_catalog_length,
            )
            coverage = matched / len(query_tokens)
            phrase_boost = (
                8.0 if len(query_norm) >= 4 and query_norm in document.normalized else 0.0
            )
            raw_coverage = _term_coverage(raw_terms, document.normalized)
            core_coverage = _term_coverage(core_terms, document.normalized)
            body_matched = sum(
                hash(token) in document.body_token_hashes for token in query_tokens
            )
            body_coverage = body_matched / len(query_tokens)
            core_query_tokens = {
                token for term in core_terms for token in tokenize(term)
            }
            body_core_coverage = (
                sum(
                    hash(token) in document.body_token_hashes
                    for token in core_query_tokens
                )
                / len(core_query_tokens)
                if core_query_tokens
                else 0.0
            )
            body_core_complete = bool(core_query_tokens) and body_core_coverage == 1.0
            path_norm = normalize_text(document.path)
            path_boost = self.exact_term_boost * sum(
                normalize_text(term) in path_norm for term in raw_terms
            )
            score = (
                bm25 * (0.6 + coverage)
                + phrase_boost
                + path_boost
                + self.exact_term_boost * raw_coverage
                + 2.0 * self.exact_term_boost * core_coverage
                + 4.0 * body_coverage
                + 6.0 * body_core_coverage
                + 20.0 * body_core_complete
            )
            document_scores[document_id] = score
            if score <= 0:
                continue
            ranked_documents.append((-score, document.path, document_id))
            if document.directory:
                directory_scores[document.directory] = max(
                    score, directory_scores.get(document.directory, 0.0)
                )

        ranked_documents.sort()
        top_directories = {
            directory
            for directory, _ in sorted(
                directory_scores.items(), key=lambda item: (-item[1], item[0])
            )[: self.route_directory_count]
        }
        routed: list[int] = []
        # 先保留全局最匹配文档，再利用目录路由补充同专业目录中的文档。
        for _, _, document_id in ranked_documents:
            if document_id not in routed:
                routed.append(document_id)
            if len(routed) >= max(1, self.route_document_count // 2):
                break
        for _, _, document_id in ranked_documents:
            if self._documents[document_id].directory in top_directories:
                if document_id not in routed:
                    routed.append(document_id)
                if len(routed) >= self.route_document_count:
                    break
        for _, _, document_id in ranked_documents:
            if document_id not in routed:
                routed.append(document_id)
            if len(routed) >= self.route_document_count:
                break
        return set(routed), document_scores

    def _chunk_score(
        self,
        chunk: _IndexedChunk,
        query_tokens: set[str],
        query_norm: str,
        raw_terms: set[str],
        core_terms: set[str],
        *,
        global_candidate: bool = False,
        doc_freq: Counter[str] | None = None,
        corpus_size: int | None = None,
        average_length: float | None = None,
    ) -> tuple[float, float] | None:
        bm25, matched = self._bm25(
            chunk.term_counts,
            chunk.length,
            query_tokens,
            doc_freq if doc_freq is not None else self._chunk_freq,
            corpus_size if corpus_size is not None else len(self._chunks),
            average_length if average_length is not None else self._avg_chunk_length,
        )
        if matched == 0:
            return None
        coverage = matched / len(query_tokens)
        phrase_boost = 7.0 if len(query_norm) >= 4 and query_norm in chunk.normalized else 0.0
        term_coverage = _term_coverage(raw_terms, chunk.normalized)
        core_coverage = _term_coverage(core_terms, chunk.normalized)
        path_norm = normalize_text(chunk.path)
        path_core_coverage = _term_coverage(core_terms, path_norm)
        has_core_term = not core_terms or core_coverage > 0 or path_core_coverage > 0
        required_matches = min(self.min_matched_tokens, len(query_tokens))
        relevant = (
            len(query_tokens) == 1
            or phrase_boost > 0
            or (
                matched >= required_matches
                and (
                    coverage >= self.min_query_coverage
                    or term_coverage >= self.min_raw_term_coverage
                )
            )
        )
        if self.require_core_term and core_terms and not has_core_term:
            relevant = False
        if (
            global_candidate
            and raw_terms
            and phrase_boost <= 0
            and term_coverage < self.global_min_raw_term_coverage
        ):
            relevant = False
        if not relevant:
            return None
        exact_term_score = self.exact_term_boost * term_coverage
        core_score = 2.0 * self.exact_term_boost * max(
            core_coverage, path_core_coverage
        )
        path_boost = 0.75 * sum(
            normalize_text(term) in path_norm for term in raw_terms
        )
        score = (
            bm25 * (0.5 + coverage)
            + phrase_boost
            + exact_term_score
            + core_score
            + path_boost
        )
        return score, coverage

    def _full_document_chunks(self, document_id: int) -> list[_IndexedChunk]:
        """按需加载候选文档的全部 chunk，并以有限 LRU 缓存避免反复读盘。"""
        cached = self._route_chunk_cache.pop(document_id, None)
        if cached is not None:
            self._route_chunk_cache[document_id] = cached
            return cached

        document = self._documents[document_id]
        path = self.root.joinpath(*PurePosixPath(document.path).parts)
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return []
        chunks: list[_IndexedChunk] = []
        for chunk_text, location in _split_document_with_locations(
            text,
            chunk_chars=self.chunk_chars,
            overlap_chars=self.overlap_chars,
        ):
            counts = Counter(tokenize(chunk_text))
            if not counts:
                continue
            chunks.append(
                _IndexedChunk(
                    document_id=document_id,
                    path=document.path,
                    location=location,
                    text=chunk_text,
                    normalized=normalize_text(chunk_text),
                    term_counts=counts,
                    length=sum(counts.values()),
                )
            )
        self._route_chunk_cache[document_id] = chunks
        while len(self._route_chunk_cache) > self.route_cache_documents:
            self._route_chunk_cache.popitem(last=False)
        return chunks

    def document_candidates(
        self, query: str, *, top_k: int = 3
    ) -> list[tuple[str, str, str]]:
        """返回文档路由候选：(安全引用、专业目录、标题)。"""
        query = query.strip()
        query_tokens = _expanded_query_tokens(query)
        if not query_tokens or not self._documents:
            return []
        query_norm = normalize_text(query)
        raw_terms, core_terms = _query_terms(query)
        routed, scores = self._route_documents(
            query, query_tokens, query_norm, raw_terms, core_terms
        )
        ranked = sorted(
            (document_id for document_id in routed if scores.get(document_id, 0.0) > 0),
            key=lambda document_id: (-scores[document_id], self._documents[document_id].path),
        )
        candidates: list[tuple[str, str, str]] = []
        for document_id in ranked[: max(1, top_k)]:
            document = self._documents[document_id]
            directory = document.directory.rsplit("/", 1)[-1] if document.directory else ""
            file_name = PurePosixPath(document.path).stem
            title = (
                f"{file_name}｜{document.title}"
                if document.title and document.title != file_name
                else file_name
            )
            candidates.append((f"D{document_id}", directory, title))
        return candidates

    def read_document_context(
        self,
        reference: str,
        query: str,
        *,
        before: int = 2,
        after: int = 2,
        max_chars: int = 1800,
    ) -> tuple[str, str] | None:
        """在候选文档内放宽 chunk 门槛定位最佳位置，再读取邻近正文。"""
        match = re.fullmatch(r"D(\d+)", reference.strip(), re.I)
        if not match:
            return None
        document_id = int(match.group(1))
        if document_id < 0 or document_id >= len(self._documents):
            return None
        chunks = self._full_document_chunks(document_id)
        query_tokens = _expanded_query_tokens(query)
        if not chunks or not query_tokens:
            return None
        frequencies: Counter[str] = Counter()
        total_length = 0
        for chunk in chunks:
            frequencies.update(chunk.term_counts.keys())
            total_length += chunk.length
        average_length = total_length / len(chunks)
        query_norm = normalize_text(query)
        ranked: list[tuple[float, int]] = []
        for chunk_index, chunk in enumerate(chunks):
            bm25, matched = self._bm25(
                chunk.term_counts,
                chunk.length,
                query_tokens,
                frequencies,
                len(chunks),
                average_length,
            )
            if matched <= 0:
                continue
            coverage = matched / len(query_tokens)
            phrase_boost = 7.0 if query_norm and query_norm in chunk.normalized else 0.0
            ranked.append((-(bm25 * (0.5 + coverage) + phrase_boost), chunk_index))
        if not ranked:
            return None
        _, chunk_index = min(ranked)
        resolved_ref = f"D{document_id}:C{chunk_index}"
        context = self.read_context(
            resolved_ref,
            before=before,
            after=after,
            max_chars=max_chars,
        )
        return (resolved_ref, context) if context else None

    def reference_for_result(self, source: str, snippet: str) -> str | None:
        """为检索结果生成稳定且不暴露路径的引用编号。"""
        path = source.rsplit(" [", 1)[0]
        document_id = next((i for i, doc in enumerate(self._documents) if doc.path == path), None)
        if document_id is None:
            return None
        snippet_norm = normalize_text(snippet.replace("［片段起始已截断］", "").replace("［片段截断］", ""))
        best_index, best_ratio = None, -1.0
        for chunk_index, chunk in enumerate(self._full_document_chunks(document_id)):
            if snippet_norm and (snippet_norm in chunk.normalized or chunk.normalized in snippet_norm):
                return f"D{document_id}:C{chunk_index}"
            ratio = SequenceMatcher(None, snippet_norm, chunk.normalized).ratio()
            if ratio > best_ratio:
                best_index, best_ratio = chunk_index, ratio
        return None if best_index is None else f"D{document_id}:C{best_index}"

    def read_context(self, reference: str, *, before: int = 2, after: int = 2, max_chars: int = 1800) -> str | None:
        """读取引用位置附近的有限 chunk，不允许路径或任意文件访问。"""
        match = re.fullmatch(r"D(\d+):C(\d+)", reference.strip(), re.I)
        if not match:
            return None
        document_id, chunk_index = (int(value) for value in match.groups())
        if document_id < 0 or document_id >= len(self._documents):
            return None
        chunks = self._full_document_chunks(document_id)
        if chunk_index < 0 or chunk_index >= len(chunks):
            return None
        start = max(0, chunk_index - max(0, before))
        end = min(len(chunks), chunk_index + max(0, after) + 1)
        parts, remaining = [], max(200, max_chars)
        for chunk in chunks[start:end]:
            text = _SPACE_RE.sub(" ", chunk.text).strip()
            if not text:
                continue
            if len(text) > remaining:
                text = text[:remaining].rstrip() + "［上下文截断］"
            parts.append(text)
            remaining -= len(text)
            if remaining <= 0:
                break
        return "\n\n".join(parts) or None

    @staticmethod
    def _snippet(chunk: str, query: str, snippet_chars: int) -> str:
        compact = _SPACE_RE.sub(" ", chunk).strip()
        if len(compact) <= snippet_chars:
            return compact
        lower = compact.lower()
        candidates = [query.strip()] + sorted(
            [part for part in _RAW_TERM_SPLIT_RE.split(query) if len(part) >= 2],
            key=len,
            reverse=True,
        )
        pos = next(
            (lower.find(part.lower()) for part in candidates if lower.find(part.lower()) >= 0),
            0,
        )
        start = max(0, pos - snippet_chars // 3)
        end = min(len(compact), start + snippet_chars)
        start = max(0, end - snippet_chars)
        excerpt = compact[start:end]
        # 尽量在完整句子处收尾，避免半句话和省略号诱导模型续写文档。
        if end < len(compact):
            endings = [match.end() for match in _SENTENCE_END_RE.finditer(excerpt)]
            usable = [position for position in endings if position >= len(excerpt) * 0.55]
            if usable:
                excerpt = excerpt[: usable[-1]]
            else:
                excerpt = excerpt.rstrip() + "［片段截断］"
        if start:
            excerpt = "［片段起始已截断］" + excerpt
        return excerpt

    def search(
        self, query: str, *, top_k: int = 3, snippet_chars: int = 400
    ) -> list[tuple[str, str]]:
        """执行文档路由、双路段落召回和统一重排。"""
        query = query.strip()
        query_tokens = _expanded_query_tokens(query)
        if not query_tokens or not self._documents:
            return []
        query_norm = normalize_text(query)
        raw_terms, core_terms = _query_terms(query)
        routed_documents, document_scores = self._route_documents(
            query, query_tokens, query_norm, raw_terms, core_terms
        )

        routed_chunks = [
            chunk
            for document_id in sorted(routed_documents)
            for chunk in self._full_document_chunks(document_id)
        ]
        routed_freq: Counter[str] = Counter()
        routed_total_length = 0
        for chunk in routed_chunks:
            routed_freq.update(chunk.term_counts.keys())
            routed_total_length += chunk.length
        routed_average_length = (
            routed_total_length / len(routed_chunks) if routed_chunks else 1.0
        )

        ranked_routed: list[tuple[float, float, str, int, _IndexedChunk]] = []
        for ordinal, chunk in enumerate(routed_chunks):
            scored = self._chunk_score(
                chunk,
                query_tokens,
                query_norm,
                raw_terms,
                core_terms,
                doc_freq=routed_freq,
                corpus_size=len(routed_chunks),
                average_length=routed_average_length,
            )
            if scored is None:
                continue
            chunk_score, coverage = scored
            final_score = (
                chunk_score
                + 0.35 * document_scores.get(chunk.document_id, 0.0)
                + 0.75
            )
            ranked_routed.append(
                (-final_score, -coverage, chunk.path, ordinal, chunk)
            )

        ranked_global: list[tuple[float, float, str, int, _IndexedChunk]] = []
        for chunk_id, chunk in enumerate(self._chunks):
            scored = self._chunk_score(
                chunk,
                query_tokens,
                query_norm,
                raw_terms,
                core_terms,
                global_candidate=True,
            )
            if scored is None:
                continue
            chunk_score, coverage = scored
            document_score = document_scores.get(chunk.document_id, 0.0)
            final_score = chunk_score + 0.35 * document_score
            ranked_global.append(
                (-final_score, -coverage, chunk.path, chunk_id, chunk)
            )

        ranked_routed.sort()
        ranked_global.sort()
        candidate_pool = (
            ranked_routed[: self.rerank_pool_size]
            + ranked_global[: self.global_supplement_count]
        )
        best_by_chunk: dict[
            tuple[str, str, str],
            tuple[float, float, str, int, _IndexedChunk],
        ] = {}
        for item in candidate_pool:
            chunk = item[4]
            key = (chunk.path, chunk.location, chunk.normalized)
            if key not in best_by_chunk or item[:4] < best_by_chunk[key][:4]:
                best_by_chunk[key] = item

        results: list[tuple[str, str]] = []
        per_file: Counter[str] = Counter()
        seen: set[tuple[str, str]] = set()
        seen_locations: set[tuple[str, str]] = set()
        for _, _, _, _, chunk in sorted(
            best_by_chunk.values(), key=lambda item: item[:4]
        ):
            if per_file[chunk.path] >= self.per_file_limit:
                continue
            location_key = (chunk.path, chunk.location)
            if chunk.location and location_key in seen_locations:
                continue
            snippet = self._snippet(chunk.text, query, max(80, snippet_chars))
            if is_low_quality_snippet(snippet):
                continue
            key = (chunk.path, normalize_text(snippet[:160]))
            if key in seen:
                continue
            if any(_near_duplicate(snippet, prior) for _, prior in results):
                continue
            seen.add(key)
            if chunk.location:
                seen_locations.add(location_key)
            per_file[chunk.path] += 1
            source = (
                f"{chunk.path} [{chunk.location}]"
                if chunk.location
                else chunk.path
            )
            results.append((source, snippet))
            if len(results) >= max(1, top_k):
                break
        return results
