'''Setup check for LedgerLens: reads real samples, builds a vector index and searches it.

Needs internet ONCE (the embedding model is about 90 MB). Needs NO API key.
Run from the project folder:   python tests/smoke_ledger_pipeline.py
'''
from __future__ import annotations

import sys
import tempfile
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, 'reconfigure'):  # never crash on old Windows console encodings
    sys.stdout.reconfigure(errors='replace')

SAMPLES = ROOT / 'sample_documents'
SHEET_ROWS_USED = 1500   # enough near-identical rows to hide the policy text, but quick to embed
MIN_PASSAGES = 1000


def stage(number: int, text: str) -> None:
    print(f'\n{number}/4 {text}')


def ok(text: str) -> None:
    print(f'   [OK] {text}')


def fail(text: str) -> None:
    print(f'   [FAILED] {text}')
    sys.exit(1)


def read(name: str):
    from ingestion import read_document
    return read_document(name, (SAMPLES / name).read_bytes())


def main() -> int:
    from ingestion import split_into_chunks
    stage(1, 'Reading one text sample and one spreadsheet sample ...')
    policy = read('Corporate_Expense_Policy.pdf')
    memo = read('Q3_Audit_Memo.txt')
    claims = read('Travel_Expense_Claims.csv')
    sheet = read('Q4_Financial_Budget.xlsx')
    ok(f'policy PDF: {len(policy)} pages with text | memo: {len(memo)} | claims CSV: {len(claims)} rows '
       f'| sales sheet: {len(sheet)} rows')
    chunks = split_into_chunks(policy + memo + claims + sheet[:SHEET_ROWS_USED])
    if len(chunks) <= MIN_PASSAGES:
        fail(f'only {len(chunks)} passages; expected more than {MIN_PASSAGES}')
    ok(f'{len(chunks)} passages (the first {SHEET_ROWS_USED} sheet rows are used so this check stays quick)')

    stage(2, 'Loading the local embedding model and indexing into a temporary ChromaDB ...')
    from retriever import (EmbeddingModelError, get_embeddings, index_chunks, initialize_vector_store,
                           semantic_search)
    started = time.perf_counter()
    try:
        get_embeddings()
    except EmbeddingModelError as exc:
        fail(str(exc))
    ok(f'embedding model ready in {time.perf_counter() - started:.1f} s')
    with tempfile.TemporaryDirectory(prefix='ledgerlens_smoke_', ignore_cleanup_errors=True) as folder:
        store = initialize_vector_store('docs_' + uuid.uuid4().hex, Path(folder))
        started = time.perf_counter()
        stored = index_chunks(store, chunks)
        ok(f'{stored} passages embedded and stored in {time.perf_counter() - started:.1f} s')

        stage(3, 'Semantic search: two needle-in-a-haystack questions and one ordinary question ...')
        checks = [
            ('What is the daily limit for meal reimbursement during university travel?', '$64'),
            # The needle names the exact row (date + region + state), not just any row of that agent.
            ('What were the sales and the target for agent Geeta in Jharkhand on 2015-07-20?',
             'Date: 2015-07-20 00:00:00 Region: North State Name: Jharkhand'),
            ('Within how many working days must a cash advance be accounted for after a trip?', '15 working days'),
        ]
        for question, needle in checks:
            hits = semantic_search(store, question, k=5)
            texts = [' '.join(hit.document.page_content.split()) for hit in hits]
            if not any(needle in text for text in texts):
                fail(f'"{question}" did not retrieve a passage containing "{needle}"')
            rank = next(i for i, text in enumerate(texts, 1) if needle in text)
            ok(f'found "{needle}" at rank {rank} of {len(hits)} for: {question}')

        stage(4, 'Guardrail: an off-topic question must be refused before any LLM is called ...')
        from agent import LedgerAgent, NO_ANSWER_TEXT
        from llm import Provider

        def never_called(provider):
            raise AssertionError('The language model must not be called for an off-topic question.')

        doc_agent = LedgerAgent([Provider('gemini', 'Gemini', 'not-a-real-model', 'not-a-real-key')],
                                model_factory=never_called)
        answer = doc_agent.answer('Who won the football world cup in 2022?', store)
        if answer.text != NO_ANSWER_TEXT or not answer.note:
            fail('the off-topic question was not refused locally with the fixed message')
        ok('off-topic question refused locally, no language model was called')
    print('\nALL CHECKS PASSED')
    return 0


if __name__ == '__main__':
    sys.exit(main())
