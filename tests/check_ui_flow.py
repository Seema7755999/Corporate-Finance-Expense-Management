'''Walk through the LedgerLens page the way an evaluator would, using Streamlit's AppTest.

Needs: the keys in .env, internet, and patience (indexing the 4 sample files embeds about 10,700 passages
on the CPU). Run from the project folder:   python tests/check_ui_flow.py
Options:  --skip-live       do not call any language model (skips the answer checks)
          --fallback-only   quick run: index only the audit memo, then test the wrong-key fallback
'''
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, 'reconfigure'):  # never crash on old Windows console encodings
    sys.stdout.reconfigure(errors='replace')

from dotenv import load_dotenv

load_dotenv(ROOT / '.env')

from streamlit.testing.v1 import AppTest

BAD_FILES = ROOT / 'sample_documents' / 'error_test_files'
SKIP_LIVE = '--skip-live' in sys.argv
FALLBACK_ONLY = '--fallback-only' in sys.argv
SAMPLE_MEMO = ROOT / 'sample_documents' / 'Q3_Audit_Memo.txt'
FAILURES: list[str] = []


def check(condition: bool, text: str) -> None:
    print(f'   [{"OK" if condition else "FAILED"}] {text}')
    if not condition:
        FAILURES.append(text)


def step(title: str) -> None:
    print(f'\n== {title}')


def texts(at: AppTest) -> str:
    '''Every piece of visible text on the page, in one string, for simple "is it there" checks.'''
    parts = []
    for kind in ('markdown', 'caption', 'success', 'info', 'warning', 'error', 'text'):
        parts += [str(el.value) for el in at.get(kind)]
    return '\n'.join(parts)


def sidebar_texts(at: AppTest) -> str:
    parts = []
    for kind in ('markdown', 'caption', 'success', 'info', 'warning', 'error'):
        parts += [str(el.value) for el in at.sidebar.get(kind)]
    return '\n'.join(parts)


def last_answer(at: AppTest) -> str:
    messages = [m for m in at.chat_message if m.name == 'assistant']
    if not messages:
        return ''
    parts = []
    for kind in ('markdown', 'caption', 'text'):
        parts += [str(el.value) for el in messages[-1].get(kind)]
    return '\n'.join(parts)


def ask(at: AppTest, question: str) -> AppTest:
    at.chat_input[0].set_value(question).run()
    return at


def click(at: AppTest, key: str) -> AppTest:
    at.button(key=key).click().run()
    return at


def button(at: AppTest, label: str):
    return next(b for b in at.button if b.label == label)


def fallback_check(at: AppTest) -> None:
    step('9. A wrong Gemini key falls back to Groq')
    os.environ.pop('PRIMARY_LLM', None)   # default order: Gemini first
    time.sleep(45)  # let Groq's per-minute token allowance recover after the earlier questions
    at.sidebar.text_input(key='typed_gemini_key').set_value('this-is-a-wrong-key').run()
    side = sidebar_texts(at)
    check('Gemini: the API key was not accepted' in side, 'the sidebar shows "Gemini: the API key was not accepted"')
    check('Active model: **Groq' in side and 'spare in use' in side, 'the sidebar shows a Groq model as the active spare')
    ask(at, 'Which department closed its corporate card reconciliation late, and by how long?')
    answer = last_answer(at)
    print('   answer:', answer[:300].replace('\n', ' | '))
    check('Marketing' in answer, 'the answer names Marketing')
    check('Answered by **Groq' in answer and 'spare model' in answer, 'the answer says it came from a Groq spare model')
    at.sidebar.text_input(key='typed_gemini_key').set_value('').run()
    check('Active model: **Gemini**' in sidebar_texts(at), 'clearing the box brings Gemini back')


def finish(started: float) -> int:
    print(f'\nFinished in {time.perf_counter() - started:.0f} s.')
    if FAILURES:
        print(f'\n{len(FAILURES)} CHECK(S) FAILED:')
        for text in FAILURES:
            print('  -', text)
        return 1
    print('\nALL UI CHECKS PASSED')
    return 0


def main() -> int:
    started = time.perf_counter()
    at = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=900)

    step('1. First load: the page renders and the model status shows')
    at.run()
    check(not at.exception, 'no exception on the first run')
    side = sidebar_texts(at)
    check('1. Language model' in [h.value for h in at.sidebar.header], 'sidebar section "1. Language model" is present')
    check('2. Add documents' in [h.value for h in at.sidebar.header], 'sidebar section "2. Add documents" is present')
    check('3. Sample questions' in [h.value for h in at.sidebar.header], 'sidebar section "3. Sample questions" is present')
    check('Active model:' in side, 'the language-model status box shows an active model')
    print('   status shown:', ' | '.join(line for line in side.splitlines() if 'model' in line.lower() or 'key' in line.lower())[:300])

    if FALLBACK_ONLY:
        step('Quick run: index only the audit memo')
        at.file_uploader[0].set_value([(SAMPLE_MEMO.name, SAMPLE_MEMO.read_bytes(), 'text/plain')])
        button(at, 'Index uploaded files').click().run()
        check(at.session_state['passage_total'] > 0, 'the audit memo is indexed')
        fallback_check(at)
        return finish(started)

    step('2. Asking before any document is indexed')
    ask(at, 'What is the daily meal limit?')
    check('Add documents first' in last_answer(at), 'the "Add documents first" message appears')

    step('3. Uploading all 7 bad files at once (plus nothing else)')
    uploads = [(p.name, p.read_bytes(), 'application/octet-stream') for p in sorted(BAD_FILES.iterdir())]
    at.file_uploader[0].set_value(uploads)
    button(at, 'Index uploaded files').click().run()
    check(not at.exception, 'no exception while handling the bad files')
    side = sidebar_texts(at)
    for fragment, label in (
        ("'fake_report.docx' cannot be used", 'fake_report.docx is refused'),
        ('Word documents are not accepted', 'the Word hint is shown'),
        ("'sample_archive.zip' cannot be used", 'sample_archive.zip is refused'),
        ('Archives such as ZIP cannot be opened here', 'the archive hint is shown'),
        ("'old_excel.xls' cannot be used", 'old_excel.xls is refused'),
        ("'corrupted.pdf': This PDF could not be opened", 'corrupted.pdf gets the PDF message'),
        ("'not_really_excel.xlsx': This Excel file could not be opened", 'not_really_excel.xlsx gets the Excel message'),
        ("'empty.txt' is empty (0 bytes), so it was skipped.", 'empty.txt is skipped as empty'),
        ('Added 3 passages from 1 file(s): accented_names_windows1252.csv', 'the valid Windows-1252 CSV is still indexed (3 passages)'),
    ):
        check(fragment in side, label)
    check(at.session_state['passage_total'] == 3, 'the knowledge base holds exactly the 3 valid passages')

    step('4. The same file twice, and "Index" with nothing chosen')
    button(at, 'Index uploaded files').click().run()
    check("'accented_names_windows1252.csv' is already indexed" in sidebar_texts(at), 'a repeated file is reported as already indexed')
    at.file_uploader[0].set_value(None)
    button(at, 'Index uploaded files').click().run()
    check('Pick at least one file first.' in sidebar_texts(at), 'indexing with nothing chosen shows the warning')

    if not SKIP_LIVE:
        step('5. A question about the Windows-1252 CSV (live model)')
        os.environ['PRIMARY_LLM'] = 'groq'   # spare the small Gemini free quota during bulk checks
        ask(at, 'Where does René Müller live?')
        answer = last_answer(at)
        print('   answer:', answer[:300].replace('\n', ' | '))
        check('Zürich' in answer, 'the answer says Zürich')

    step('6. Clear the knowledge base')
    button(at, 'Clear knowledge base').click().run()
    check(at.session_state['passage_total'] == 0, 'the passage count is back to 0')
    check('Knowledge base emptied' in texts(at), 'the "Knowledge base emptied" banner appears')

    step('7. Load the 4 sample documents (this is the slow step)')
    t0 = time.perf_counter()
    button(at, 'Load sample finance documents').click().run()
    seconds = time.perf_counter() - t0
    side = sidebar_texts(at)
    print(f'   indexing took {seconds:.0f} s')
    check(not at.exception, 'no exception while indexing the samples')
    check('from 4 file(s)' in side and 'Added' in side, 'a green box says "Added ... passages from 4 file(s)"')
    print('   report:', [line for line in side.splitlines() if line.startswith('Added')][:1])
    check(at.session_state['passage_total'] > 1000, f"more than 1000 passages indexed ({at.session_state['passage_total']})")

    if not SKIP_LIVE:
        step('8. Every sample-question button produces a chat answer')
        os.environ['PRIMARY_LLM'] = 'groq'
        for key in ('ok_0', 'ok_1', 'ok_2', 'ok_3', 'no_0', 'no_1', 'no_2'):
            label = at.button(key=key).label
            click(at, key)
            answer = last_answer(at)
            check(bool(answer.strip()), f'{key}: a chat answer appeared for "{label[:60]}"')
            print('      ->', answer[:260].replace('\n', ' | '))
            time.sleep(20)  # be gentle with the free tier's tokens-per-minute limit

        fallback_check(at)

    step('10. Clear again')
    button(at, 'Clear knowledge base').click().run()
    check(at.session_state['passage_total'] == 0, 'the knowledge base is empty again')

    return finish(started)


if __name__ == '__main__':
    sys.exit(main())
