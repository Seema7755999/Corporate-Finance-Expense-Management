'''Reading uploaded files: type checks, safe loading of PDF / TXT / CSV / XLSX, and chunking.

Everything that can go wrong with a user's file ends in a short, friendly DocumentRejectedError
message. Raw library errors never reach the screen.
'''
from __future__ import annotations

import hashlib
import logging
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Iterator

import pandas as pd
from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

logger = logging.getLogger('ledgerlens')
logging.getLogger('pypdf').setLevel(logging.ERROR)  # pypdf warns loudly about harmless PDF quirks

# ----------------------------------------------------------------------------- limits
ALLOWED_SUFFIXES = {'.pdf', '.txt', '.csv', '.xlsx'}
ALLOWED_TYPES_NOTE = 'Allowed file types: PDF, TXT, CSV and XLSX.'
MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_EXTRACTED_CHARACTERS = 2_000_000
MAX_RECORDS = 20_000
MAX_WORKBOOK_UNZIPPED_BYTES = 100 * 1024 * 1024
UPLOAD_DIR = Path(__file__).resolve().parent / 'uploaded_documents'

_SESSION_ID = re.compile(r'docs_[a-f0-9]{32}')

# ----------------------------------------------------------------------------- friendly texts
_ARCHIVE = ('Archives such as ZIP cannot be opened here. Unpack the archive first, then upload '
            'the PDF, TXT, CSV or XLSX files that were inside it.')
_WORD = ('Word documents are not accepted. In Word choose File > Save As and pick PDF (or Plain Text), '
         'then upload that file instead.')
_PICTURE = ('Pictures cannot be read because this app has no text recognition (OCR). Upload a PDF or '
            'TXT file that holds real text.')
_SLIDES = 'PowerPoint files are not accepted. Export the slides to PDF and upload the PDF instead.'
_OLD_EXCEL = ('The old Excel format (.xls) is not accepted. Open the file in Excel, choose '
              'Save As > Excel Workbook (.xlsx) or CSV, and upload that.')
REFUSAL_HINTS = {
    '.docx': _WORD, '.doc': _WORD, '.odt': _WORD, '.rtf': _WORD,
    '.zip': _ARCHIVE, '.rar': _ARCHIVE, '.7z': _ARCHIVE, '.gz': _ARCHIVE, '.tar': _ARCHIVE,
    '.xls': _OLD_EXCEL, '.xlsm': _OLD_EXCEL, '.pptx': _SLIDES, '.ppt': _SLIDES,
    '.png': _PICTURE, '.jpg': _PICTURE, '.jpeg': _PICTURE, '.gif': _PICTURE, '.bmp': _PICTURE,
    '.tif': _PICTURE, '.tiff': _PICTURE, '.webp': _PICTURE,
    '.exe': 'Programs are not documents, so they cannot be opened here.',
}
BROKEN_FILE_HINTS = {
    '.pdf': 'This PDF could not be opened. It may be damaged or may not be a real PDF.',
    '.xlsx': ('This Excel file could not be opened. It may be damaged, protected by a password, '
              'or not a real .xlsx file.'),
    '.csv': 'This CSV file could not be read. It may be damaged or may be binary data with a .csv name.',
    '.txt': 'This text file could not be read. It may be damaged or may be binary data with a .txt name.',
}


class DocumentRejectedError(ValueError):
    '''A file cannot be used. The message is written for the end user.'''


def clean_filename(name: str) -> str:
    '''Reduce a client-supplied file name to a harmless final component (never trust upload names).'''
    last = str(name).replace('\\', '/').split('/')[-1]
    last = re.sub(r'[^\w.\- ()]', '_', last).lstrip('.')
    return last[:120] or 'document'


# ----------------------------------------------------------------------------- Excel
class WorkbookRowLoader(BaseLoader):
    '''Turns every non-empty spreadsheet row into one Document of "Column: value" lines.

    A small custom loader on purpose: the heavy `unstructured` package is not needed for this.
    '''

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def _check_unzipped_size(self) -> None:
        with zipfile.ZipFile(self.path) as archive:
            if sum(item.file_size for item in archive.infolist()) > MAX_WORKBOOK_UNZIPPED_BYTES:
                raise DocumentRejectedError('The unpacked Excel workbook is larger than 100 MiB.')

    def lazy_load(self) -> Iterator[Document]:
        self._check_unzipped_size()
        workbook = pd.ExcelFile(self.path, engine='openpyxl')
        for sheet in workbook.sheet_names:
            frame = pd.read_excel(workbook, sheet_name=sheet, dtype=str, keep_default_na=False,
                                  nrows=MAX_RECORDS + 1)
            if len(frame) > MAX_RECORDS:
                raise DocumentRejectedError('An Excel sheet has more than 20,000 rows, which is over the limit.')
            columns = [str(column) for column in frame.columns]
            # Header is spreadsheet row 1, so the first data row is row 2 (what the user sees in Excel).
            for row_number, values in enumerate(frame.itertuples(index=False), start=2):
                lines = [f'{column}: {str(value).strip()}'
                         for column, value in zip(columns, values) if str(value).strip()]
                if lines:
                    yield Document(page_content='\n'.join(lines),
                                   metadata={'sheet': str(sheet), 'row': row_number})


# ----------------------------------------------------------------------------- text helpers
def _text_encoding(data: bytes) -> str | None:
    '''Best guess for a text file's encoding, or None when it looks like binary data.'''
    if data.startswith((b'\xff\xfe', b'\xfe\xff')):
        return 'utf-16'
    if b'\x00' in data:
        return None
    for encoding in ('utf-8-sig', 'cp1252'):  # cp1252 is what Excel writes when it saves a CSV
        try:
            data.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    return None


def _broken_file_message(suffix: str, path: Path) -> str:
    if suffix == '.pdf':
        try:
            if PdfReader(path).is_encrypted:
                return 'This PDF is protected by a password. Remove the password and upload it again.'
        except Exception:  # noqa: BLE001 - the file is already known to be unreadable; just fall through
            pass
    return BROKEN_FILE_HINTS[suffix]


def explain_empty(filename: str, size: int) -> str:
    '''Why a file produced no passages, phrased to follow the file name in a sentence.'''
    if size == 0:
        return 'is empty (0 bytes)'
    suffix = Path(filename).suffix.lower()
    if suffix == '.pdf':
        return 'contains no selectable text (it may be a scan, and OCR is not supported)'
    if suffix in {'.csv', '.xlsx'}:
        return 'has no data rows'
    return 'contains only blank text'


def _loader_for(suffix: str, path: Path, encoding: str | None) -> BaseLoader:
    if suffix == '.pdf':
        return PyPDFLoader(str(path))
    if suffix == '.txt':
        return TextLoader(str(path), encoding=encoding)
    if suffix == '.csv':
        return CSVLoader(str(path), encoding=encoding)
    return WorkbookRowLoader(path)


def _tidy_metadata(raw: dict, suffix: str, safe_name: str, file_id: str) -> dict:
    '''Keep only the few keys the rest of the app understands; drop temp paths and loader internals.'''
    meta: dict = {'source': safe_name, 'file_id': file_id, 'format': suffix.lstrip('.')}
    if suffix == '.pdf' and isinstance(raw.get('page'), int):
        meta['page'] = raw['page'] + 1  # PyPDFLoader counts pages from 0; people count from 1
    if suffix == '.csv' and isinstance(raw.get('row'), int):
        meta['row'] = raw['row'] + 2    # CSVLoader counts data rows from 0; the sheet header is row 1
    if suffix == '.xlsx':
        if 'row' in raw:
            meta['row'] = raw['row']
        if 'sheet' in raw:
            meta['sheet'] = raw['sheet']
    return meta


def read_document(filename: str, data: bytes) -> list[Document]:
    '''Validate and read one uploaded file into Documents (pages for PDF, rows for tables).

    Returns [] for a zero-byte file. Raises DocumentRejectedError with a friendly message otherwise.
    '''
    safe_name = clean_filename(filename)
    suffix = Path(safe_name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        if not suffix:
            reason = 'This file has no extension.'
        else:
            reason = REFUSAL_HINTS.get(suffix, f"'{suffix}' files are not supported.")
        raise DocumentRejectedError(f"'{safe_name}' cannot be used. {reason} {ALLOWED_TYPES_NOTE}")
    if not data:
        return []
    if len(data) > MAX_FILE_BYTES:
        raise DocumentRejectedError(f"'{safe_name}' is bigger than the 20 MiB limit for one file.")

    encoding = None
    if suffix in {'.txt', '.csv'}:
        encoding = _text_encoding(data)
        if encoding is None:
            raise DocumentRejectedError(f"'{safe_name}': {BROKEN_FILE_HINTS[suffix]}")

    file_id = hashlib.sha256(data).hexdigest()
    # A fixed temp name (never the client's) keeps odd names away from the loaders and avoids
    # Windows file-locking problems with files left open in the upload folder.
    with tempfile.TemporaryDirectory(prefix='ledgerlens_', ignore_cleanup_errors=True) as folder:
        path = Path(folder) / f'upload{suffix}'
        try:
            path.write_bytes(data)
            documents: list[Document] = []
            characters = 0
            for document in _loader_for(suffix, path, encoding).lazy_load():
                if not document.page_content.strip():
                    continue
                characters += len(document.page_content)
                if characters > MAX_EXTRACTED_CHARACTERS or len(documents) >= MAX_RECORDS:
                    raise DocumentRejectedError(
                        f"'{safe_name}' has more than 2 million characters or 20,000 records, "
                        'which is over the limit.')
                document.metadata = _tidy_metadata(document.metadata, suffix, safe_name, file_id)
                documents.append(document)
            return documents
        except DocumentRejectedError:
            raise
        except Exception as exc:  # noqa: BLE001 - any loader failure becomes one friendly message
            logger.warning('Could not read %s: %s', safe_name, type(exc).__name__)
            # Must run before the temp folder disappears: it re-opens the file to spot a password.
            raise DocumentRejectedError(f"'{safe_name}': {_broken_file_message(suffix, path)}") from exc


# ----------------------------------------------------------------------------- chunking
def split_into_chunks(documents: list[Document], chunk_size: int = 700,
                      chunk_overlap: int = 100) -> list[Document]:
    '''Cut documents into small passages. 700 characters keep MiniLM (about 256 word pieces) from truncating.'''
    if chunk_size <= 0:
        raise ValueError('chunk_size must be positive.')
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError('chunk_overlap must be at least 0 and smaller than chunk_size.')
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True,
        separators=['\n\n', '\n', ' ', ''])
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        # Same file + same position + same text always gives the same id, so re-indexing never duplicates.
        key = f"{chunk.metadata.get('file_id', '')}:{index}:{chunk.page_content}"
        chunk.metadata['chunk_id'] = hashlib.sha256(key.encode('utf-8')).hexdigest()
    return chunks


# ----------------------------------------------------------------------------- uploaded_documents/
def _session_folder(session_id: str, upload_dir: Path | None = None) -> Path:
    if not _SESSION_ID.fullmatch(session_id or ''):
        raise ValueError('Invalid session id.')
    return (upload_dir or UPLOAD_DIR) / session_id


def store_upload_copy(session_id: str, filename: str, data: bytes, upload_dir: Path | None = None) -> Path:
    '''Keep a copy of an uploaded file in uploaded_documents/<session>/ (under its cleaned name).'''
    folder = _session_folder(session_id, upload_dir)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / clean_filename(filename)
    target.write_bytes(data)
    return target


def remove_session_copies(session_id: str, upload_dir: Path | None = None) -> None:
    '''Delete the stored copies of one browser session.'''
    shutil.rmtree(_session_folder(session_id, upload_dir), ignore_errors=True)


def clear_old_uploads(upload_dir: Path | None = None) -> None:
    '''Remove everything in uploaded_documents/ except .gitkeep (leftovers of earlier runs).'''
    folder = upload_dir or UPLOAD_DIR
    if not folder.exists():
        return
    for item in folder.iterdir():
        if item.name == '.gitkeep':
            continue
        try:
            shutil.rmtree(item) if item.is_dir() else item.unlink()
        except OSError:  # another running copy may still hold the file; leaving it is harmless
            logger.info('Could not remove %s (in use?)', item.name)
