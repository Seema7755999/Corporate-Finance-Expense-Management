'''The searchable memory of the app: local MiniLM embeddings plus a ChromaDB vector store.

Nothing here calls an online service except the one-time download of the embedding model.
'''
from __future__ import annotations

import logging
import os
import re
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

# Quiet, offline-friendly behaviour. These must be set BEFORE chromadb / huggingface are imported.
os.environ.setdefault('ANONYMIZED_TELEMETRY', 'False')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
os.environ.setdefault('HF_HUB_VERBOSITY', 'error')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

from langchain_chroma import Chroma  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402

logger = logging.getLogger('ledgerlens')
logging.getLogger('huggingface_hub').setLevel(logging.ERROR)

DEFAULT_DB_PATH = Path(__file__).resolve().parent / 'chroma_db'
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
# Chroma's stock HNSW index (ef_search=100) is approximate. In a collection that holds a few policy
# paragraphs among thousands of near-identical table rows it can miss the truly best passage.
# Collections here are small, so a thorough search costs only milliseconds. The measured effect on
# the sample documents is reported in DOCUMENTATION.md (section 9).
HNSW_SETTINGS = {'space': 'cosine', 'max_neighbors': 64, 'ef_construction': 800, 'ef_search': 2000}

BATCH_SIZE = 128
_COLLECTION_NAME = re.compile(r'docs_[a-f0-9]{32}')


class EmbeddingModelError(RuntimeError):
    '''The local embedding model could not be loaded (usually: no internet on the very first start).'''


@dataclass(frozen=True)
class SearchHit:
    '''One retrieved passage and how close it is to the question (cosine similarity, -1 to 1).'''
    document: Document
    similarity: float


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    '''Load the MiniLM model once per process. The first ever call downloads about 90 MB.'''
    try:
        model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu', 'trust_remote_code': False},
            encode_kwargs={'normalize_embeddings': True})
        # PyTorch defaults to the number of physical cores; using every logical CPU measured about
        # 40% faster indexing on a 2-core / 4-thread laptop.
        import torch
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        return model
    except Exception as exc:  # noqa: BLE001 - network, disk or library failures all mean the same thing here
        logger.warning('Embedding model failed to load: %s', type(exc).__name__)
        raise EmbeddingModelError(
            'The local embedding model could not be loaded. The very first start downloads it '
            '(about 90 MB), so it needs an internet connection once. Check your connection or '
            'firewall and start the app again.') from exc


def initialize_vector_store(collection_name: str, persist_directory: Path = DEFAULT_DB_PATH) -> Chroma:
    '''Open (or create) one Chroma collection. Names look like docs_<32 hex>, one per browser session.'''
    if not _COLLECTION_NAME.fullmatch(collection_name or ''):
        raise ValueError('Invalid collection name.')
    Path(persist_directory).mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=collection_name,
        persist_directory=str(persist_directory),
        embedding_function=get_embeddings(),
        collection_configuration={'hnsw': dict(HNSW_SETTINGS)})


def purge_stale_vector_data(path: Path = DEFAULT_DB_PATH) -> None:
    '''Empty chroma_db/ (keeping .gitkeep): vector data from earlier runs is never reused.'''
    folder = Path(path)
    if not folder.exists():
        return
    for item in folder.iterdir():
        if item.name == '.gitkeep':
            continue
        try:
            shutil.rmtree(item) if item.is_dir() else item.unlink()
        except OSError:  # another running copy may hold these files; leaving them is harmless
            logger.info('Could not remove %s (in use?)', item.name)


def index_chunks(store: Chroma, chunks: list[Document],
                 on_progress: Callable[[int, int], None] | None = None) -> int:
    '''Embed and store chunks in batches. Ids come from chunk_id, so indexing twice never duplicates.'''
    total = len(chunks)
    done = 0
    for start in range(0, total, BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        store.add_documents(batch, ids=[chunk.metadata['chunk_id'] for chunk in batch])
        done += len(batch)
        if on_progress is not None:
            on_progress(done, total)
    return done


def semantic_search(store: Chroma, question: str, k: int = 5, min_similarity: float = 0.30) -> list[SearchHit]:
    '''Return the best passages for a question, most similar first, dropping weak matches.'''
    if not 1 <= k <= 20:
        raise ValueError('k must be between 1 and 20.')
    if not -1 <= min_similarity <= 1:
        raise ValueError('min_similarity must be between -1 and 1.')
    if not question or not question.strip():
        return []
    hits: list[SearchHit] = []
    for document, distance in store.similarity_search_with_score(question.strip(), k=k):
        # Chroma reports cosine DISTANCE (0 = identical). Convert explicitly instead of trusting
        # any integration's default "relevance" scaling.
        similarity = max(-1.0, min(1.0, 1.0 - float(distance)))
        if similarity >= min_similarity:
            hits.append(SearchHit(document=document, similarity=similarity))
    return hits
