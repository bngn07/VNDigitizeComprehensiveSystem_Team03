from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from openai import OpenAI

logger = logging.getLogger('module_4_2')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s - %(message)s')

MODEL      = 'qwen2.5:3b'
API_BASE   = 'http://localhost:11434/v1'
MAX_TOKENS = 1024
MAX_WORDS  = 2000
LOW_CONF   = 0.8

LENGTH_INSTRUCTIONS = {
    'ngan':       '1-2 câu ngắn gọn, chỉ nêu chủ đề và quyết định chính',
    'trung_binh': '3-5 câu, nêu chủ thể ban hành, nội dung chính, đối tượng thực hiện',
    'chi_tiet':   '1 đoạn đầy đủ (6-10 câu): căn cứ pháp lý, nội dung cụ thể, trách nhiệm và hiệu lực',
}

SYSTEM_PROMPT = (
    'Bạn là AI phân tích văn bản hành chính Việt Nam. '
    'Chỉ trả về JSON thuần túy, không markdown, không giải thích.\n'
    'Văn bản có thể chứa lỗi OCR — hãy suy luận từ ngữ cảnh.\n'
    'classification: Quyết định | Công văn | Bản án | Biên bản | Hợp đồng | Thông tư | Nghị định | Chỉ thị | Đơn/Kiến nghị | Khác\n'
    'JSON trả về:\n'
    '{\n'
    '  "classification": "<loại văn bản>",\n'
    '  "summary": "<trích yếu>",\n'
    '  "keywords": ["từ khóa 1", "từ khóa 2"],\n'
    '  "confidence_overall": <0.0 - 1.0>\n'
    '}'
)

_SUMMARY_PATTERNS = [
    re.compile(r'[Tt]rích\s+yếu\s*[:\-]\s*(.+?)(?=\n\n|\Z)',        re.DOTALL),
    re.compile(r'[Tt]RÍ[Cc][Hh]\s+[Yy]ẾU\s*[:\-]\s*(.+?)(?=\n\n|\Z)', re.DOTALL),
    re.compile(r'[Tt]óm\s+tắt\s*[:\-]\s*(.+?)(?=\n\n|\Z)',          re.DOTALL),
]


# Data models

class SummaryLength(str, Enum):
    NGAN       = 'ngan'
    TRUNG_BINH = 'trung_binh'
    CHI_TIET   = 'chi_tiet'


class DocType(str, Enum):
    QUYET_DINH    = 'Quyết định'
    CONG_VAN      = 'Công văn'
    BIEN_BAN      = 'Biên bản'
    HOP_DONG      = 'Hợp đồng'
    BAN_AN        = 'Bản án'
    THONG_TU      = 'Thông tư'
    NGHI_DINH     = 'Nghị định'
    CHI_THI       = 'Chỉ thị'
    DON_KIEN_NGHI = 'Đơn/Kiến nghị'
    KHAC          = 'Khác'
    TU_DONG       = 'Tự động nhận diện'


@dataclass
class BoundingBox:
    x1: int = 0; y1: int = 0; x2: int = 0; y2: int = 0
    def to_dict(self): return {'x1': self.x1, 'y1': self.y1, 'x2': self.x2, 'y2': self.y2}


@dataclass
class MetadataField:
    field_name: str
    value: str
    confidence: float
    bounding_box: Optional[BoundingBox] = None

    def to_dict(self):
        return {
            'field_name':   self.field_name,
            'value':        self.value,
            'confidence':   round(self.confidence, 4),
            'bounding_box': self.bounding_box.to_dict() if self.bounding_box else None,
        }


@dataclass
class SubDocument:
    type: str; page_start: int; page_end: int
    def to_dict(self): return {'type': self.type, 'page_start': self.page_start, 'page_end': self.page_end}


@dataclass
class SummaryResult:
    document_id:        str
    classification:     str
    summary:            str
    confidence_overall: float
    keywords:           list[str]           = field(default_factory=list)
    metadata:           list[MetadataField] = field(default_factory=list)
    sub_documents:      list[SubDocument]   = field(default_factory=list)
    has_existing_summary:    bool  = False
    ocr_confidence:          float = 0.0
    flagged_word_count:      int   = 0
    processing_time_seconds: float = 0.0
    input_word_count:        int   = 0
    summary_word_count:      int   = 0
    model_used:              str   = MODEL
    error:                   Optional[str] = None

    def to_api_dict(self):
        return {
            'document_id':        self.document_id,
            'classification':     self.classification,
            'summary':            self.summary,
            'confidence_overall': round(self.confidence_overall, 4),
            'keywords':           self.keywords,
            'metadata':           [m.to_dict() for m in self.metadata],
            'sub_documents':      [s.to_dict() for s in self.sub_documents],
        }

    def to_api_json(self, indent: int = 2):
        return json.dumps(self.to_api_dict(), ensure_ascii=False, indent=indent)

    def to_debug_dict(self):
        return {
            **self.to_api_dict(),
            'has_existing_summary':    self.has_existing_summary,
            'ocr_confidence':          self.ocr_confidence,
            'flagged_word_count':      self.flagged_word_count,
            'processing_time_seconds': self.processing_time_seconds,
            'input_word_count':        self.input_word_count,
            'summary_word_count':      self.summary_word_count,
            'model_used':              self.model_used,
            'error':                   self.error,
        }

    def to_debug_json(self, indent: int = 2):
        return json.dumps(self.to_debug_dict(), ensure_ascii=False, indent=indent)


# Exceptions

class SummarizerError(Exception): pass
class InputTooShortError(SummarizerError): pass
class LowOCRQualityError(SummarizerError): pass
class APIError(SummarizerError): pass


# Helpers

def _extract_from_ocr_result(ocr_result) -> tuple[str, float, int, list[str]]:
    if hasattr(ocr_result, 'words') and hasattr(ocr_result, 'raw_text'):
        raw_text = ocr_result.raw_text or ''
        conf     = float(ocr_result.overall_confidence or 0.0)
        flagged  = [w.text for w in ocr_result.words if getattr(w, 'flagged', False)]
        return raw_text, conf, len(flagged), flagged
    if hasattr(ocr_result, 'texts'):
        texts = ocr_result.texts
        if isinstance(texts, str):
            return texts, 0.0, 0, []
        raw_text = ' '.join(b.text for b in texts)
        conf     = float(ocr_result.confidence)
        flagged  = [b.text for b in texts if b.confidence < LOW_CONF]
        return raw_text, conf, len(flagged), flagged
    raise TypeError(f'Không nhận ra kiểu OCRResult: {type(ocr_result)}')


_OCR_NORMALIZE = [
    (re.compile(r'\bTHẮM\b'),       'THẨM'),
    (re.compile(r'\bTHẦM\b'),       'THẨM'),
    (re.compile(r'\bDẦN\b'),        'DÂN'),
    (re.compile(r'\bĐÂN\b'),        'DÂN'),
    (re.compile(r'\bTÓI\b'),        'TỐI'),
    (re.compile(r'\bTỎI\b'),        'TỐI'),
    (re.compile(r'\bTHẰNH\b'),      'THÀNH'),
    (re.compile(r'\bPHÓ\b'),        'PHỐ'),
    (re.compile(r'\bHỎ\b'),         'HỒ'),
    (re.compile(r'\bHÔ\b'),         'HỒ'),
    (re.compile(r'\bNHÂN\s+DẦN\b'), 'NHÂN DÂN'),
    (re.compile(r'\bTỈNH\s+BN\b'),  'TỈNH BÌNH DƯƠNG'),
]


def _normalize_ocr_text(text: str) -> str:
    for pattern, replacement in _OCR_NORMALIZE:
        text = pattern.sub(replacement, text)
    return text


def _convert_metadata(module3_output: Optional[dict]) -> list[MetadataField]:
    if not module3_output:
        return []
    def _to_field(item: dict) -> Optional[MetadataField]:
        if not item.get('value'):
            return None
        value = str(item['value'])
        if item.get('field_name') == 'co_quan_ban_hanh':
            value = _normalize_ocr_text(value)
        bb_raw = item.get('bounding_box')
        return MetadataField(
            field_name=item['field_name'], value=value,
            confidence=float(item.get('confidence', 0.0)),
            bounding_box=BoundingBox(**bb_raw) if bb_raw else None,
        )
    return [f for item in module3_output.get('metadata', []) if (f := _to_field(item))]


def _detect_existing_summary(text: str) -> Optional[str]:
    for pattern in _SUMMARY_PATTERNS:
        m = pattern.search(text)
        if m:
            summary = m.group(1).strip()
            if len(summary.split()) >= 5:
                return summary
    return None


def _build_prompt(text: str, doc_type: DocType, length: SummaryLength,
                  flagged_words: list[str], ocr_conf: float) -> str:
    type_hint    = '' if doc_type == DocType.TU_DONG else f'Gợi ý loại văn bản: {doc_type.value}\n'
    length_instr = LENGTH_INSTRUCTIONS[length.value]
    quality_note = ''
    if ocr_conf < LOW_CONF:
        quality_note = f'\nOCR confidence thấp ({ocr_conf:.1%}). Suy luận từ ngữ cảnh.\n'
    if flagged_words:
        quality_note += f'Từ bị flag: {", ".join(flagged_words[:10])}\n'
    words   = text.split()
    trimmed = ' '.join(words[:MAX_WORDS])
    if len(words) > MAX_WORDS:
        trimmed += '\n[... văn bản bị cắt bớt ...]'
    return (
        f'{type_hint}'
        f'Độ dài trích yếu: {length_instr}\n'
        f'{quality_note}\n'
        f'NỘI DUNG VĂN BẢN:\n{"─" * 60}\n{trimmed}\n{"─" * 60}'
    )


# AutoSummarizer

class AutoSummarizer:
    def __init__(self, model: str = MODEL, base_url: str = API_BASE, api_key: str = 'local'):
        self.model  = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        logger.info('AutoSummarizer | model=%s | base_url=%s', model, base_url)

    def summarize_from_ocr(self, ocr_result, doc_type=DocType.TU_DONG,
                           length=SummaryLength.NGAN, document_id=None,
                           min_ocr_confidence=0.0, module3_metadata=None,
                           sub_documents=None) -> SummaryResult:
        raw_text, ocr_conf, flagged_count, flagged_words = _extract_from_ocr_result(ocr_result)
        if min_ocr_confidence > 0 and ocr_conf < min_ocr_confidence:
            raise LowOCRQualityError(f'OCR confidence {ocr_conf:.1%} < ngưỡng {min_ocr_confidence:.1%}')
        return self._run(raw_text, doc_type, length, document_id,
                         ocr_conf, flagged_words, flagged_count,
                         module3_metadata, sub_documents or [])

    def summarize(self, text: str, doc_type=DocType.TU_DONG,
                  length=SummaryLength.NGAN, document_id=None,
                  module3_metadata=None, sub_documents=None) -> SummaryResult:
        return self._run(text.strip(), doc_type, length, document_id,
                         1.0, [], 0, module3_metadata, sub_documents or [])

    def summarize_file(self, filepath: str, **kwargs) -> SummaryResult:
        return self.summarize(self._read_file(filepath), **kwargs)

    def _run(self, text, doc_type, length, document_id,
             ocr_conf, flagged_words, flagged_count,
             module3_metadata, sub_documents) -> SummaryResult:
        text = text.strip()
        word_count = len(text.split())
        if word_count < 20:
            raise InputTooShortError(f'Văn bản quá ngắn ({word_count} từ). Cần ít nhất 20 từ.')

        doc_id   = document_id or f'doc-{int(time.time())}'
        existing = _detect_existing_summary(text)
        if existing:
            logger.info('doc_id=%s đã có trích yếu sẵn, bỏ qua LLM.', doc_id)
            return SummaryResult(
                document_id=doc_id, classification='Không xác định',
                summary=existing, confidence_overall=1.0,
                metadata=_convert_metadata(module3_metadata),
                sub_documents=sub_documents, has_existing_summary=True,
                ocr_confidence=round(ocr_conf, 4), flagged_word_count=flagged_count,
                input_word_count=word_count, summary_word_count=len(existing.split()),
                model_used='existing',
            )

        logger.info('[%s] words=%d conf=%.2f flagged=%d', doc_id, word_count, ocr_conf, flagged_count)
        t0      = time.perf_counter()
        raw     = self._call_llm(text, doc_type, length, flagged_words, ocr_conf)
        elapsed = round(time.perf_counter() - t0, 3)
        parsed  = self._parse(raw)

        summary_text  = parsed.get('summary', '')
        llm_conf      = float(parsed.get('confidence_overall', 0.8))
        flagged_ratio = flagged_count / max(len(text.split()), 1)
        conf          = round(min(llm_conf, ocr_conf) * (1 - flagged_ratio * 0.3), 4)

        result = SummaryResult(
            document_id=doc_id,
            classification=parsed.get('classification', 'Không xác định'),
            summary=summary_text,
            confidence_overall=conf,
            keywords=parsed.get('keywords') or [],
            metadata=_convert_metadata(module3_metadata),
            sub_documents=sub_documents,
            has_existing_summary=False,
            ocr_confidence=round(ocr_conf, 4),
            flagged_word_count=flagged_count,
            processing_time_seconds=elapsed,
            input_word_count=word_count,
            summary_word_count=len(summary_text.split()),
            model_used=self.model,
        )
        logger.info('[%s] done %.2fs conf=%.2f', doc_id, elapsed, result.confidence_overall)
        return result

    def _call_llm(self, text, doc_type, length, flagged_words, ocr_conf) -> str:
        prompt = _build_prompt(text, doc_type, length, flagged_words, ocr_conf)
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user',   'content': prompt},
                ],
                max_tokens=MAX_TOKENS,
                temperature=0.1,
            )
            return resp.choices[0].message.content or ''
        except Exception as e:
            raise APIError(f'Không gọi được LLM ({self.client.base_url}). Lỗi: {e}') from e

    def _parse(self, raw: str) -> dict:
        cleaned = raw.strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            m = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group())
                except json.JSONDecodeError:
                    pass
            logger.warning('Không parse được JSON, dùng fallback.')
            return {'classification': 'Không xác định', 'summary': cleaned[:500],
                    'keywords': [], 'confidence_overall': 0.5}

    def _read_file(self, filepath: str) -> str:
        ext = os.path.splitext(filepath)[1].lower()
        if ext == '.txt':
            with open(filepath, encoding='utf-8') as f:
                return f.read()
        if ext == '.pdf':
            try:
                import pdfplumber
            except ImportError:
                raise ImportError('pip install pdfplumber')
            pages = []
            with pdfplumber.open(filepath) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        pages.append(t)
            if not pages:
                raise SummarizerError('Không trích xuất được text từ PDF.')
            return '\n\n'.join(pages)
        raise ValueError(f'Không hỗ trợ định dạng: {ext}')


# Batch

def batch_summarize(items: list[dict], summarizer: AutoSummarizer,
                    default_length=SummaryLength.NGAN,
                    workers: int = 1) -> list[SummaryResult]:
    def _process(item: dict, i: int) -> tuple[int, SummaryResult]:
        doc_id = item.get('document_id', f'batch-doc-{i + 1}')
        try:
            r = summarizer.summarize_from_ocr(
                ocr_result=item['ocr_result'],
                doc_type=item.get('doc_type', DocType.TU_DONG),
                length=item.get('length', default_length),
                document_id=doc_id,
                module3_metadata=item.get('module3_metadata'),
                sub_documents=item.get('sub_documents', []),
            )
        except (SummarizerError, KeyError, TypeError) as e:
            logger.error('Batch %s lỗi: %s', doc_id, e)
            r = SummaryResult(document_id=doc_id, classification='',
                              summary='', confidence_overall=0.0, error=str(e))
        return i, r

    results = [None] * len(items)
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_process, item, i): i for i, item in enumerate(items)}
            for future in as_completed(futures):
                idx, r = future.result()
                results[idx] = r
    else:
        for i, item in enumerate(items):
            _, r = _process(item, i)
            results[i] = r
    return results


# CLI output helpers

def _sep(char: str = '-', width: int = 60) -> str:
    return char * width


def _wrap(text: str, width: int = 56) -> list[str]:
    words, line, lines = text.split(), '', []
    for w in words:
        candidate = f'{line} {w}'.strip()
        if len(candidate) > width:
            if line:
                lines.append(line)
            line = w
        else:
            line = candidate
    if line:
        lines.append(line)
    return lines


def _print_result(result: SummaryResult, index: int, total: int, debug: bool = False) -> None:
    W = 60
    print()
    print(_sep('=', W))
    print(f'  [{index}/{total}] {result.document_id}')
    print(_sep('-', W))
    print(f'  {"Phan loai":<20} {result.classification}')
    print(f'  {"Do tin cay":<20} {result.confidence_overall:.0%}'
          + (f'  (OCR: {result.ocr_confidence:.0%})' if result.ocr_confidence < 1.0 else ''))
    print(f'  {"So tu":<20} {result.input_word_count}  ->  trich yeu: {result.summary_word_count} tu')
    if debug:
        print(f'  {"Thoi gian":<20} {result.processing_time_seconds:.3f}s')
        print(f'  {"Model":<20} {result.model_used}')
        if result.flagged_word_count:
            print(f'  {"Flagged words":<20} {result.flagged_word_count}')
    print(_sep('-', W))
    for ln in _wrap(result.summary):
        print(f'  {ln}')
    if result.keywords:
        print(_sep('-', W))
        print(f'  Keywords: {", ".join(result.keywords)}')
    if result.error:
        print(_sep('-', W))
        print(f'  ERROR: {result.error}')
    print(_sep('=', W))


def _print_batch_summary(results: list[SummaryResult], elapsed: float) -> None:
    W        = 60
    ok       = [r for r in results if not r.error]
    failed   = [r for r in results if r.error]
    avg_conf = sum(r.confidence_overall for r in ok) / len(ok) if ok else 0.0
    avg_time = sum(r.processing_time_seconds for r in ok) / len(ok) if ok else 0.0

    print()
    print(_sep('=', W))
    print(f'  {"Total":<20} {len(results)}')
    print(f'  {"Passed":<20} {len(ok)}')
    print(f'  {"Failed":<20} {len(failed)}')
    print(f'  {"Avg confidence":<20} {avg_conf:.0%}')
    print(f'  {"Avg time":<20} {avg_time:.2f}s / doc')
    print(f'  {"Total time":<20} {elapsed:.2f}s')
    if failed:
        print(_sep('-', W))
        for r in failed:
            print(f'  {r.document_id}: {r.error}')
    print(_sep('=', W))
    print()


# CLI entry point

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='summarizer',
        description='Module 4.2 — Auto-Summarization (Local LLM / Ollama)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python summarizer.py --input van_ban.txt --length trung_binh
  python summarizer.py --inputs a.txt b.txt c.pdf --workers 2
  python summarizer.py --inputs a.txt b.txt --workers 2 --output ketqua.json
  python summarizer.py --input van_ban.txt --model gemini-2.5-flash-lite
        """,
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--input',  metavar='FILE', help='Mot file dau vao (.txt hoac .pdf)')
    src.add_argument('--inputs', metavar='FILE', nargs='+', help='Nhieu file dau vao (chay batch)')

    p.add_argument('--length',   choices=['ngan', 'trung_binh', 'chi_tiet'], default='trung_binh')
    p.add_argument('--doc-type', dest='doc_type', choices=[e.value for e in DocType],
                   default=DocType.TU_DONG.value)
    p.add_argument('--model',    default=MODEL)
    p.add_argument('--base-url', dest='base_url', default=API_BASE)
    p.add_argument('--workers',  type=int, default=1)
    p.add_argument('--output',   metavar='FILE')
    p.add_argument('--debug',    action='store_true')
    p.add_argument('--json-only', action='store_true', dest='json_only')

    return p


def main() -> None:
    parser = _build_arg_parser()
    args   = parser.parse_args()

    length   = SummaryLength(args.length)
    doc_type = DocType(args.doc_type)
    files    = [args.input] if args.input else args.inputs

    summ    = AutoSummarizer(model=args.model, base_url=args.base_url)
    t_start = time.perf_counter()

    if len(files) == 1:
        result = summ.summarize_file(files[0], length=length, doc_type=doc_type)
        if args.json_only:
            print(result.to_debug_json() if args.debug else result.to_api_json())
        else:
            _print_result(result, 1, 1, debug=args.debug)
        if args.output:
            data = result.to_debug_dict() if args.debug else result.to_api_dict()
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f'\n  Saved -> {args.output}')
        return

    items = [
        {
            'ocr_result':  type('_R', (), {
                'raw_text':           summ._read_file(fp),
                'overall_confidence': 1.0,
                'words':              [],
            })(),
            'doc_type':    doc_type,
            'length':      length,
            'document_id': os.path.basename(fp),
        }
        for fp in files
    ]

    results = batch_summarize(items, summ, default_length=length, workers=args.workers)
    elapsed = round(time.perf_counter() - t_start, 3)

    if args.json_only:
        output_data = [r.to_debug_dict() if args.debug else r.to_api_dict() for r in results]
        print(json.dumps(output_data, ensure_ascii=False, indent=2))
    else:
        for i, result in enumerate(results, 1):
            _print_result(result, i, len(results), debug=args.debug)
        _print_batch_summary(results, elapsed)

    if args.output:
        output_data = [r.to_debug_dict() if args.debug else r.to_api_dict() for r in results]
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f'  Saved -> {args.output}\n')


if __name__ == '__main__':
    main()