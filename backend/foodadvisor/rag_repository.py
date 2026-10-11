from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


SUPPORTED_SUFFIXES = {".json", ".md", ".pdf", ".txt"}
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_BATCH_SIZE = 32
INDEX_VERSION = "complete-recipe-chunks-v7"
RECIPE_TIME_LINE_RE = re.compile(
    r"^\s*\d+(?:[.,]\d+)?\s*(?:min(?:uto)?s?\.?|seg(?:undo)?s?\.?|h(?:ora)?s?\.?)"
    r"(?:\s+\d+(?:[.,]\d+)?\s*(?:min(?:uto)?s?\.?|seg(?:undo)?s?\.?|h(?:ora)?s?\.?))?\s*$",
    re.IGNORECASE,
)
RECIPE_HEADER_TIME_RE = re.compile(
    r"^\s*\d+(?:[.,]\d+)?\s*(?:min(?:uto)?s?\.?|seg(?:undo)?s?\.?|h(?:ora)?s?\.?)"
    r"(?:\s+\d+(?:[.,]\d+)?\s*(?:min(?:uto)?s?\.?|seg(?:undo)?s?\.?|h(?:ora)?s?\.?))?"
    r"\s*\|",
    re.IGNORECASE,
)
RECIPE_CONTENT_TIME_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:min(?:uto)?s?\.?|seg(?:undo)?s?\.?|h(?:ora)?s?\.?)\b",
    re.IGNORECASE,
)
RECIPE_QUANTITY_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:kg|g|mg|ml|cl|dl|l|"
    r"colheres?|chávenas?|xícaras?|ovos?|unidades?|dentes?|latas?|"
    r"fatias?|ramos?|folhas?|embalagens?)\b",
    re.IGNORECASE,
)
RECIPE_ACTION_RE = re.compile(
    r"\b(?:junte|adicione|misture|coloque|leve|bata|aqueça|aqueca|cozinhe|"
    r"sirva|reserve|triture|prepare|amasse|polvilhe|despeje|retire|"
    r"préaqueça|preaqueça|pre-aqueça|enrole|corte|pese)\b",
    re.IGNORECASE,
)
RECIPE_CONTINUATION_CUE_RE = re.compile(
    r"\b(?:forno|cozedura|cozer|assar|ferver|lume|massa|tabuleiro|"
    r"frigideira|temperatura|receita|preparacao)\b",
    re.IGNORECASE,
)


class RAGUnavailableError(RuntimeError):
    """The recipe knowledge base cannot serve a retrieval request."""


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


@dataclass
class RecipeChunk:
    source: str
    title: str
    content: str
    chunk: int
    page: int | None = None


class FastEmbedder:
    def __init__(self, model_name: str) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise RAGUnavailableError(
                "Dependências RAG em falta. Instale backend/requirements.txt."
            ) from exc

        try:
            self.model = TextEmbedding(model_name=model_name)
        except (OSError, ValueError) as exc:
            raise RAGUnavailableError(
                f"Não foi possível carregar o modelo de embeddings '{model_name}'."
            ) from exc

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[start : start + EMBEDDING_BATCH_SIZE]
            vectors.extend(
                self.model.embed(batch, batch_size=EMBEDDING_BATCH_SIZE, parallel=None)
            )

        matrix = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-12)


class RecipeRAGRepository:
    """Loads recipe resources, indexes their chunks as vectors, and retrieves evidence."""

    def __init__(
        self,
        resources_dir: Path | None = None,
        index_dir: Path | None = None,
        embedder: Embedder | None = None,
        model_name: str | None = None,
        top_k: int = 4,
        min_similarity: float = 0.30,
    ) -> None:
        project_root = Path(__file__).resolve().parents[2]
        self.resources_dir = resources_dir or Path(
            os.environ.get("FOODADVISOR_RESOURCES_DIR", project_root / "resources")
        )
        self.index_dir = index_dir or Path(
            os.environ.get("FOODADVISOR_RAG_INDEX_DIR", project_root / "backend" / ".rag_index")
        )
        self.model_name = model_name or os.environ.get(
            "FOODADVISOR_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
        )
        self.embedder = embedder
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.chunks: list[RecipeChunk] = []
        self.vectors: np.ndarray | None = None
        self._indexed = False
        self._indexed_fingerprint: str | None = None

    def search(self, recipe_text: str) -> list[dict]:
        self._ensure_index()
        if self.vectors is None or not self.chunks:
            return []

        query_vector = self._get_embedder().encode([recipe_text])[0]
        similarities = self.vectors @ query_vector
        normalized_query = self._normalize_search_text(recipe_text)
        query_terms = {
            term
            for term in normalized_query.split()
            if len(term) > 2 and term not in {"com", "para", "por", "uma", "uns", "das", "dos"}
        }
        ranked_results = []
        for index, similarity in enumerate(similarities):
            normalized_content = self._normalize_search_text(self.chunks[index].content)
            term_coverage = (
                sum(term in normalized_content.split() for term in query_terms) / len(query_terms)
                if query_terms
                else 0.0
            )
            exact_phrase = bool(normalized_query and normalized_query in normalized_content)
            rank_score = float(similarity) + 0.15 * term_coverage + 0.35 * exact_phrase
            if similarity >= self.min_similarity or exact_phrase:
                ranked_results.append(
                    (rank_score, int(index), float(similarity), exact_phrase)
                )

        ranked_results.sort(reverse=True)
        exact_matches = [result for result in ranked_results if result[3]]
        if exact_matches:
            best_results = [
                (index, similarity)
                for _, index, similarity, _ in exact_matches[: self.top_k]
            ]
        else:
            best_results = [
                (index, similarity)
                for _, index, similarity, _ in ranked_results[: self.top_k]
            ]

        return [
            {
                **asdict(self.chunks[index]),
                "score": similarity,
            }
            for index, similarity in best_results
        ]

    def format_context(self, documents: list[dict]) -> str:
        if not documents:
            return "Não foi recuperada evidência relevante na pasta de recursos."

        return "\n\n".join(
            f"[Fonte: {document['source']} | secção: {document['title']} | "
            + (f"página: {document['page']} | " if document.get("page") else "")
            + f"similaridade: {document['score']:.3f}]\n{document['content']}"
            for document in documents
        )

    def rebuild_index(self) -> dict[str, int | list[str]]:
        files = self._resource_files()
        if not files:
            raise RAGUnavailableError(
                f"Não existem recursos de receitas em {self.resources_dir}. "
                "Adicione ficheiros .txt, .md, .json ou .pdf e tente novamente."
            )

        try:
            fingerprint = self._fingerprint(files)
        except OSError as exc:
            raise RAGUnavailableError(
                "Não foi possível ler os recursos para reconstruir o índice RAG."
            ) from exc

        chunks = self._load_chunks(files)
        vectors = self._get_embedder().encode(
            [f"{chunk.title}\n{chunk.content}" for chunk in chunks]
        )
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise RAGUnavailableError(
                "O modelo de embeddings devolveu um índice incompatível com as receitas."
            )

        self._save_cached_index(fingerprint, chunks, vectors)
        self.chunks = chunks
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self._indexed_fingerprint = fingerprint
        self._indexed = True
        indexed_sources = {chunk.source for chunk in chunks}
        ignored_resources = [
            path.relative_to(self.resources_dir).as_posix()
            for path in files
            if path.relative_to(self.resources_dir).as_posix() not in indexed_sources
        ]
        return {
            "resource_files": len(files),
            "recipe_chunks": len(chunks),
            "ignored_resources": ignored_resources,
        }

    def _ensure_index(self) -> None:
        files = self._resource_files()
        if not files:
            raise RAGUnavailableError(
                f"Não existem recursos de receitas em {self.resources_dir}. "
                "Adicione ficheiros .txt, .md, .json ou .pdf e tente novamente."
            )

        try:
            fingerprint = self._fingerprint(files)
        except OSError as exc:
            raise RAGUnavailableError(
                "Não foi possível ler os recursos para construir o índice RAG."
            ) from exc
        if self._indexed and fingerprint == self._indexed_fingerprint:
            return
        if self._load_cached_index(fingerprint):
            self._indexed_fingerprint = fingerprint
            self._indexed = True
            return

        chunks = self._load_chunks(files)
        vectors = self._get_embedder().encode(
            [f"{chunk.title}\n{chunk.content}" for chunk in chunks]
        )
        self._save_cached_index(fingerprint, chunks, vectors)
        self.chunks = chunks
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self._indexed_fingerprint = fingerprint
        self._indexed = True

    def _resource_files(self) -> list[Path]:
        if not self.resources_dir.is_dir():
            raise RAGUnavailableError(
                f"A pasta de recursos não existe: {self.resources_dir}"
            )

        return sorted(
            path
            for path in self.resources_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_SUFFIXES
            and path.name.lower() != "readme.md"
        )

    def _load_chunks(self, files: list[Path]) -> list[RecipeChunk]:
        chunks: list[RecipeChunk] = []
        for path in files:
            relative_source = path.relative_to(self.resources_dir).as_posix()
            if path.suffix.lower() == ".pdf":
                recipes = self._read_pdf_resource(path)
                for index, (title, content, page_number) in enumerate(recipes):
                    chunks.append(
                        RecipeChunk(
                            source=relative_source,
                            title=title,
                            content=content,
                            chunk=index,
                            page=page_number,
                        )
                    )
            else:
                recipes = self._read_resource(path)
                for index, (title, content) in enumerate(recipes):
                    chunks.append(
                        RecipeChunk(
                            source=relative_source,
                            title=title,
                            content=content,
                            chunk=index,
                        )
                    )

        if not chunks:
            raise RAGUnavailableError(
                f"Não foi possível extrair texto dos recursos em {self.resources_dir}."
            )
        return chunks

    def _read_resource(self, path: Path) -> list[tuple[str, str]]:
        suffix = path.suffix.lower()
        try:
            if suffix == ".json":
                data = json.loads(path.read_text(encoding="utf-8"))
                recipes = self._json_recipes(data, path.stem)
            else:
                text = path.read_text(encoding="utf-8")
                recipes = (
                    self._markdown_recipes(text, path.stem)
                    if suffix == ".md"
                    else self._plain_text_recipes(text, path.stem)
                )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RAGUnavailableError(f"Não foi possível ler o recurso: {path}") from exc

        if not recipes:
            raise RAGUnavailableError(f"O recurso está vazio ou sem texto: {path}")
        return recipes

    def _read_pdf_resource(self, path: Path) -> list[tuple[str, str, int | None]]:
        try:
            from pypdf import PdfReader
            from pypdf.errors import PdfReadError
        except ImportError as exc:
            raise RAGUnavailableError(
                "Para indexar PDF, instale a dependência pypdf."
            ) from exc

        try:
            reader = PdfReader(path)
            pages = []
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text(extraction_mode="layout") or ""
                if text.strip():
                    pages.append((page_number, text))
        except (OSError, PdfReadError) as exc:
            raise RAGUnavailableError(f"Não foi possível ler o PDF: {path}") from exc

        if not pages:
            return []
        recipes = self._split_pdf_into_recipes(pages)
        if not recipes:
            return []
        return recipes

    @staticmethod
    def _is_index_page(text: str) -> bool:
        dotted_leaders = re.findall(r"(?:\.\s*){4,}\d{1,3}\b", text)
        return len(dotted_leaders) >= 5

    @staticmethod
    def _json_recipes(data: object, fallback: str) -> list[tuple[str, str]]:
        if isinstance(data, list) and data and all(
            isinstance(item, dict) and (item.get("name") or item.get("title"))
            for item in data
        ):
            return [
                (
                    str(item.get("name") or item.get("title")),
                    json.dumps(item, ensure_ascii=False, indent=2),
                )
                for item in data
            ]

        if isinstance(data, dict):
            title = str(data.get("name") or data.get("title") or fallback)
        else:
            title = fallback
        return [(title, json.dumps(data, ensure_ascii=False, indent=2))]

    @staticmethod
    def _markdown_recipes(text: str, fallback: str) -> list[tuple[str, str]]:
        normalized = text.strip()
        if not normalized:
            return []

        headings = list(re.finditer(r"(?m)^#\s+(.+?)\s*#*\s*$", normalized))
        if len(headings) <= 1:
            title = headings[0].group(1).strip() if headings else fallback
            return [(title, normalized)]

        recipes = []
        for index, heading in enumerate(headings):
            end = headings[index + 1].start() if index + 1 < len(headings) else len(normalized)
            content = normalized[heading.start() : end].strip()
            recipes.append((heading.group(1).strip(), content))
        return recipes

    @staticmethod
    def _plain_text_recipes(text: str, fallback: str) -> list[tuple[str, str]]:
        normalized = text.strip()
        if not normalized:
            return []

        lines = normalized.splitlines()
        headings = []
        for line_index, line in enumerate(lines):
            title = line.strip()
            if not title or len(title) > 100 or title.startswith(("•", "-", "#")):
                continue
            next_line = next(
                (candidate.strip() for candidate in lines[line_index + 1 :] if candidate.strip()),
                "",
            )
            if RECIPE_TIME_LINE_RE.fullmatch(next_line):
                headings.append((line_index, title))

        if not headings:
            return [(fallback, normalized)]

        recipes = []
        for index, (start, title) in enumerate(headings):
            end = headings[index + 1][0] if index + 1 < len(headings) else len(lines)
            recipes.append((title, "\n".join(lines[start:end]).strip()))
        return recipes

    @staticmethod
    def _split_pdf_into_recipes(
        pages: list[tuple[int, str]]
    ) -> list[tuple[str, str, int | None]]:
        for parser in (
            RecipeRAGRepository._split_pdf_with_time_headers,
            RecipeRAGRepository._split_ladymaxx_recipes,
        ):
            recipes = parser(pages)
            if recipes is not None:
                return recipes

        return RecipeRAGRepository._split_structured_recipe_pages(pages)

    @staticmethod
    def _split_pdf_with_time_headers(
        pages: list[tuple[int, str]]
    ) -> list[tuple[str, str, int | None]] | None:
        headings: dict[int, set[int]] = {}
        for page_number, page_text in pages:
            if RecipeRAGRepository._is_index_page(page_text):
                continue
            lines = page_text.splitlines()
            for line_index, line in enumerate(lines):
                title = RecipeRAGRepository._clean_pdf_recipe_title(line)
                if (
                    not title
                    or len(title) > 100
                    or title.startswith(("•", "-", "#"))
                    or RecipeRAGRepository._is_pdf_page_header(title)
                ):
                    continue
                next_line = next(
                    (candidate.strip() for candidate in lines[line_index + 1 :] if candidate.strip()),
                    "",
                )
                if RECIPE_HEADER_TIME_RE.match(next_line):
                    headings.setdefault(page_number, set()).add(line_index)

        if not headings:
            return None

        recipes: list[tuple[str, list[str], int]] = []
        section_index = -1
        for page_number, page_text in pages:
            if RecipeRAGRepository._is_index_page(page_text):
                continue
            lines = page_text.splitlines()
            for line_index, line in enumerate(lines):
                if line_index in headings.get(page_number, set()):
                    title = RecipeRAGRepository._clean_pdf_recipe_title(line)
                    recipes.append((title, [title], page_number))
                    section_index += 1
                elif (
                    section_index >= 0
                    and not RecipeRAGRepository._is_pdf_page_header(line.strip())
                ):
                    recipes[section_index][1].append(line)

        return [
            (
                title,
                re.sub(r"\s+", " ", " ".join(lines)).strip(),
                page_number,
            )
            for title, lines, page_number in recipes
            if " ".join(lines).strip()
        ]

    @staticmethod
    def _clean_pdf_recipe_title(line: str) -> str:
        return re.split(
            r"\s{2,}GUIA EXPRESS\b", line.strip(), maxsplit=1, flags=re.IGNORECASE
        )[0].strip()

    @staticmethod
    def _split_ladymaxx_recipes(
        pages: list[tuple[int, str]]
    ) -> list[tuple[str, str, int | None]] | None:
        recipes: list[tuple[str, list[str], int]] = []
        active_recipe: int | None = None
        found_ladymaxx_section = False

        for page_number, page_text in pages:
            if RecipeRAGRepository._is_index_page(page_text):
                continue

            lines = page_text.splitlines()
            section_starts = []
            for line_index, line in enumerate(lines):
                if not RecipeRAGRepository._is_ladymaxx_ingredients_heading(line):
                    continue

                found_ladymaxx_section = True
                title_start, title = RecipeRAGRepository._ladymaxx_title_before(
                    lines, line_index
                )
                if not title:
                    raise RAGUnavailableError(
                        "Foi detetada uma secção de receita Ladymaxx sem título "
                        f"identificável (página {page_number})."
                    )
                section_starts.append((title_start, title))

            if not section_starts:
                if (
                    active_recipe is not None
                    and RecipeRAGRepository._is_recipe_continuation(page_text)
                ):
                    recipes[active_recipe][1].extend(
                        line
                        for line in lines
                        if not RecipeRAGRepository._is_ladymaxx_page_furniture(line)
                    )
                continue

            first_start = section_starts[0][0]
            if (
                active_recipe is not None
                and RecipeRAGRepository._is_recipe_continuation(
                    "\n".join(lines[:first_start])
                )
            ):
                recipes[active_recipe][1].extend(
                    line
                    for line in lines[:first_start]
                    if not RecipeRAGRepository._is_ladymaxx_page_furniture(line)
                )

            for section_index, (start, title) in enumerate(section_starts):
                end = (
                    section_starts[section_index + 1][0]
                    if section_index + 1 < len(section_starts)
                    else len(lines)
                )
                content_lines = [
                    line
                    for line in lines[start:end]
                    if not RecipeRAGRepository._is_ladymaxx_page_furniture(line)
                ]
                recipes.append((title, content_lines, page_number))
                active_recipe = len(recipes) - 1

        if not found_ladymaxx_section:
            return None

        return [
            (
                title,
                re.sub(r"\s+", " ", " ".join(lines)).strip(),
                page_number,
            )
            for title, lines, page_number in recipes
            if " ".join(lines).strip()
        ]

    @staticmethod
    def _is_ladymaxx_ingredients_heading(line: str) -> bool:
        return RecipeRAGRepository._normalize_search_text(line).startswith(
            "ingredientes para"
        )

    @staticmethod
    def _ladymaxx_title_before(lines: list[str], heading_index: int) -> tuple[int, str]:
        title_lines = []
        index = heading_index - 1
        while index >= 0 and len(title_lines) < 3:
            line = lines[index].strip()
            if not line:
                index -= 1
                continue
            normalized = RecipeRAGRepository._normalize_search_text(line)
            if (
                len(line) > 90
                or not any(character.isalpha() for character in line)
                or RecipeRAGRepository._is_ladymaxx_page_furniture(line)
                or normalized.startswith(
                    ("ingredientes", "preparacao", "o nosso conselho", "pagina")
                )
                or line.endswith((".", "!", "?", ";"))
            ):
                break
            title_lines.append(line)
            index -= 1

        title_lines.reverse()
        return (index + 1, " ".join(title_lines))

    @staticmethod
    def _is_ladymaxx_page_furniture(line: str) -> bool:
        normalized = RecipeRAGRepository._normalize_search_text(line)
        return (
            normalized.startswith("aldi ladymaxx")
            or normalized.startswith("livro de receitas pagina")
        )

    @staticmethod
    def _split_structured_recipe_pages(
        pages: list[tuple[int, str]]
    ) -> list[tuple[str, str, int | None]]:
        recipes: list[tuple[str, list[str], int]] = []
        active_recipe: int | None = None

        for page_number, page_text in pages:
            if RecipeRAGRepository._is_index_page(page_text):
                continue

            if RecipeRAGRepository._is_structured_recipe_page(page_text):
                title = RecipeRAGRepository._recipe_page_title(page_text)
                if not title:
                    raise RAGUnavailableError(
                        f"Foi detetada uma página de receita sem título identificável "
                        f"(página {page_number})."
                    )
                recipes.append((title, [page_text], page_number))
                active_recipe = len(recipes) - 1
                continue

            if active_recipe is not None and RecipeRAGRepository._is_recipe_continuation(page_text):
                recipes[active_recipe][1].append(page_text)

        return [
            (title, re.sub(r"\s+", " ", " ".join(lines)).strip(), page_number)
            for title, lines, page_number in recipes
            if " ".join(lines).strip()
        ]

    @staticmethod
    def _is_structured_recipe_page(text: str) -> bool:
        normalized = RecipeRAGRepository._normalize_search_text(text)
        quantities = len(RECIPE_QUANTITY_RE.findall(text))
        actions = len(RECIPE_ACTION_RE.findall(normalized))
        has_recipe_section = (
            "ingredientes" in normalized
            and ("modo de preparacao" in normalized or "preparacao" in normalized)
        )
        has_serving_metadata = "porcoes" in normalized and "dificuldade" in normalized
        has_recipe_content = (
            (quantities >= 2 and actions >= 2)
            or (quantities >= 1 and actions >= 1 and (has_recipe_section or has_serving_metadata))
        )
        return has_recipe_content and (
            has_recipe_section
            or has_serving_metadata
            or RecipeRAGRepository._has_title_at_page_top(text)
        )

    @staticmethod
    def _has_title_at_page_top(text: str) -> bool:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in lines[:8]:
            letters = [character for character in line if character.isalpha()]
            if len(letters) < 3 or len(line) > 90:
                continue
            uppercase_ratio = sum(character.isupper() for character in letters) / len(letters)
            if uppercase_ratio >= 0.45:
                return True
        return False

    @staticmethod
    def _recipe_page_title(text: str) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        title_lines = []
        for line in lines[:8]:
            normalized = RecipeRAGRepository._normalize_search_text(line)
            if any(
                marker in normalized
                for marker in ("ingredientes", "porcoes", "dificuldade", "modo de preparacao")
            ):
                break
            letters = [character for character in line if character.isalpha()]
            if len(letters) < 3 or len(line) > 90:
                if title_lines:
                    break
                continue
            uppercase_ratio = sum(character.isupper() for character in letters) / len(letters)
            if uppercase_ratio >= 0.45:
                title_lines.append(line)
            elif title_lines:
                break
        return " ".join(title_lines[:3]).strip()

    @staticmethod
    def _is_recipe_continuation(text: str) -> bool:
        normalized = RecipeRAGRepository._normalize_search_text(text)
        has_method_steps = len(RECIPE_ACTION_RE.findall(normalized)) >= 2
        has_recipe_cue = bool(
            RECIPE_QUANTITY_RE.search(text)
            or RECIPE_CONTENT_TIME_RE.search(text)
            or RECIPE_CONTINUATION_CUE_RE.search(normalized)
        )
        return has_method_steps and has_recipe_cue

    @staticmethod
    def _is_pdf_page_header(line: str) -> bool:
        return bool(
            re.fullmatch(r"\d+\s+ChefEXPRESS", line, re.IGNORECASE)
            or re.fullmatch(r"[A-ZÀ-Ü\s]+\s+\d{1,3}", line)
        )

    def _fingerprint(self, files: list[Path]) -> str:
        digest = hashlib.sha256(self.model_name.encode("utf-8"))
        digest.update(INDEX_VERSION.encode("utf-8"))
        for path in files:
            digest.update(path.relative_to(self.resources_dir).as_posix().encode("utf-8"))
            digest.update(path.read_bytes())
        return digest.hexdigest()

    @staticmethod
    def _normalize_search_text(text: str) -> str:
        decomposed = unicodedata.normalize("NFKD", text.casefold())
        without_accents = "".join(
            character for character in decomposed if not unicodedata.combining(character)
        )
        return re.sub(r"[^a-z0-9]+", " ", without_accents).strip()

    def _get_embedder(self) -> Embedder:
        if self.embedder is None:
            self.embedder = FastEmbedder(self.model_name)
        return self.embedder

    def _cache_paths(self) -> tuple[Path, Path]:
        return self.index_dir / "metadata.json", self.index_dir / "vectors.npy"

    def _load_cached_index(self, fingerprint: str) -> bool:
        metadata_path, vectors_path = self._cache_paths()
        if not metadata_path.is_file() or not vectors_path.is_file():
            return False

        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict) or metadata.get("fingerprint") != fingerprint:
                return False
            vectors = np.load(vectors_path, allow_pickle=False)
            chunks = [RecipeChunk(**chunk) for chunk in metadata["chunks"]]
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return False

        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            return False
        self.chunks = chunks
        self.vectors = np.asarray(vectors, dtype=np.float32)
        return True

    def _save_cached_index(
        self,
        fingerprint: str,
        chunks: list[RecipeChunk],
        vectors: np.ndarray,
    ) -> None:
        metadata_path, vectors_path = self._cache_paths()
        self.index_dir.mkdir(parents=True, exist_ok=True)
        metadata_tmp = metadata_path.with_suffix(".tmp")
        vectors_tmp = vectors_path.with_suffix(".tmp")

        metadata_tmp.write_text(
            json.dumps(
                {
                    "fingerprint": fingerprint,
                    "chunks": [asdict(chunk) for chunk in chunks],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        with vectors_tmp.open("wb") as file:
            np.save(file, vectors, allow_pickle=False)
        metadata_tmp.replace(metadata_path)
        vectors_tmp.replace(vectors_path)
