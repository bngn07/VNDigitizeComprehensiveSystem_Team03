import os
import re
import json
import time
import math
import uuid
import logging
from dataclasses import dataclass
from typing import Optional, Type, Tuple, Any, List, Dict

from google import genai
from google.genai import types
from json_repair import repair_json
from pydantic import BaseModel
from tqdm.auto import tqdm

from .base import ExtractorResult, BaseExtractor, TextInput, T
from .prompt import Prompt
from .schema import auto_detect_schema, SCHEMA_REGISTRY
from .scoring import score_extraction

logger = logging.getLogger(__name__)


@dataclass
class GeminiParams:
    model_name: str = "gemini-2.0-flash"
    temperature: float = 0.0
    max_output_tokens: int = 512
    max_input_chars: int = 3000
    max_input_chars_retry: int = 1500
    batch_size: int = 10
    batch_poll_interval: int = 20
    batch_timeout_sec: int = 7200
    max_retries: int = 2
    retry_delay: float = 1.5
    low_confidence_threshold: float = 0.65


def clean_and_parse(raw: str, doc_id: str = "") -> Optional[dict]:
    raw = raw.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]+?)```", raw)
    candidate = m.group(1).strip() if m else raw

    s, e = candidate.find("{"), candidate.rfind("}")
    if s != -1 and e != -1:
        candidate = candidate[s : e + 1]

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    try:
        r = repair_json(candidate, return_objects=True)
        if isinstance(r, dict) and r:
            return r
    except Exception:
        pass

    if doc_id:
        logger.warning(f"[{doc_id}] parse fail | raw[:200]={repr(raw[:200])}")
    return None


def _parse_input_doc(doc: dict) -> Tuple[str, str, str]:
    doc_id = (
        doc.get("document_id")
        or doc.get("sample_id")
        or doc.get("image_name")
        or "Unknown"
    )
    if "module2_output" in doc and isinstance(doc["module2_output"], dict):
        raw = doc["module2_output"].get("raw_text", "")
    elif "ground_truth_text" in doc:
        raw = doc["ground_truth_text"]
    else:
        raw = doc.get("text", "")
    return str(doc_id), str(raw), doc.get("schema_type", "")


def _resolve_schema(doc_id: str, raw_text: str, schema_hint: str) -> Tuple[str, Type[BaseModel]]:
    if schema_hint and schema_hint in SCHEMA_REGISTRY:
        return schema_hint, SCHEMA_REGISTRY[schema_hint]
    schema_name = auto_detect_schema(raw_text)
    schema_class = SCHEMA_REGISTRY.get(schema_name)
    if schema_class is None:
        from .schema import DonTuCamKetSchema
        schema_class = DonTuCamKetSchema
    return schema_name, schema_class


def _build_result(
    doc_id: str,
    schema_name: str,
    schema_class: Type,
    raw_output: str,
    low_conf_threshold: float,
) -> ExtractorResult:
    parsed = clean_and_parse(raw_output, doc_id)
    if not parsed:
        return ExtractorResult(
            schema_used=schema_name,
            error="json_parse_failed",
            raw_output_sample=raw_output[:300],
        )

    metadata_list, overall, fill_rate = score_extraction(
        parsed, schema_class, low_conf_threshold
    )
    record_data = {m["field_name"]: m["value"] for m in metadata_list}
    try:
        record = schema_class(**record_data)
    except Exception:
        record = None

    return ExtractorResult(
        record=record,
        schema_used=schema_name,
        extracted_dynamic_data={m["field_name"]: m for m in metadata_list},
        confidence_overall=overall,
        fill_rate=fill_rate,
        raw_output_sample=raw_output[:300],
    )


class GeminiExtractor(BaseExtractor):

    def __init__(self, api_key: str, params: Optional[GeminiParams] = None):
        self.params = params or GeminiParams()
        self.client = genai.Client(api_key=api_key)
        self._gen_cfg = types.GenerateContentConfig(
            response_mime_type="application/json",
            max_output_tokens=self.params.max_output_tokens,
            temperature=self.params.temperature,
        )
        logger.info(
            f"GeminiExtractor ready | model={self.params.model_name} "
            f"batch_size={self.params.batch_size}"
        )

    def _extract_single(self, text: str, schema: Optional[Type[T]] = None) -> ExtractorResult[T]:
        doc_id = f"doc_{uuid.uuid4().hex[:8]}"

        if schema is not None:
            schema_name, schema_class = schema.__name__, schema
        else:
            schema_name, schema_class = _resolve_schema(doc_id, text, "")

        last_raw = ""
        for attempt in range(1, self.params.max_retries + 2):
            try:
                prompt = Prompt.build(
                    schema_class, text, retry=(attempt > 1),
                    max_chars=self.params.max_input_chars,
                    max_chars_retry=self.params.max_input_chars_retry,
                )
                t0 = time.time()
                resp = self.client.models.generate_content(
                    model=self.params.model_name,
                    contents=str(prompt),
                    config=self._gen_cfg,
                )
                last_raw = resp.text or ""
                elapsed = time.time() - t0

                if clean_and_parse(last_raw, doc_id):
                    logger.info(f"[{doc_id}] ✅ direct OK lần {attempt} ({elapsed:.1f}s)")
                    return _build_result(
                        doc_id, schema_name, schema_class, last_raw,
                        self.params.low_confidence_threshold,
                    )
                logger.warning(f"[{doc_id}] ⚠️  parse fail lần {attempt}")
            except Exception as exc:
                logger.error(f"[{doc_id}] ❌ lần {attempt}: {exc}")
                time.sleep(self.params.retry_delay * attempt)

        return ExtractorResult(
            schema_used=schema_name,
            error="failed_after_retries",
            raw_output_sample=last_raw[:300],
        )

    def _extract_batch(self, texts: List[str], schema: Optional[Type[T]] = None) -> List[ExtractorResult[T]]:
        docs = [{"document_id": f"doc_{i}", "text": t} for i, t in enumerate(texts)]
        if schema is not None:
            for d in docs:
                d["schema_type"] = schema.__name__
        return [self._dict_to_result(r) for r in self.run_batch_pipeline(docs)]

    def run_batch_pipeline(self, documents: List[dict]) -> List[dict]:
        n_batches = math.ceil(len(documents) / self.params.batch_size)
        logger.info(f"Tổng {len(documents)} doc → {n_batches} batch × {self.params.batch_size} doc/batch")

        submitted: List[Tuple[Optional[str], List[dict]]] = []
        for i in range(n_batches):
            chunk = documents[i * self.params.batch_size : (i + 1) * self.params.batch_size]
            requests = [r for doc in chunk if (r := self._make_batch_request(doc)) is not None]
            if not requests:
                continue
            try:
                batch = self.client.batches.create(model=self.params.model_name, src=requests)
                logger.info(f"  📤 Batch {i+1}/{n_batches} | ID: {batch.name} | {len(requests)} requests")
                submitted.append((batch.name, chunk))
                time.sleep(0.3)
            except Exception as exc:
                logger.error(f"  ❌ Batch {i+1} submit lỗi: {exc} → fallback direct")
                submitted.append((None, chunk))

        all_results: List[dict] = []
        for batch_name, chunk in tqdm(submitted, desc="Batches", unit="batch"):
            all_results.extend(self._process_one_batch(batch_name, chunk))
        return all_results

    def submit_batch_jobs(self, documents: List[dict], state_file: str = "batch_state.json") -> str:
        n_batches = math.ceil(len(documents) / self.params.batch_size)
        job_id = uuid.uuid4().hex[:12]
        logger.info(f"[submit] {len(documents)} doc → {n_batches} batch | state_file={state_file}")

        job_state: dict = {
            "job_id":       job_id,
            "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model_name":   self.params.model_name,
            "total_docs":   len(documents),
            "batches":      [],
            "direct_docs":  [],
        }

        for i in range(n_batches):
            chunk = documents[i * self.params.batch_size : (i + 1) * self.params.batch_size]
            requests = [r for doc in chunk if (r := self._make_batch_request(doc)) is not None]
            if not requests:
                continue
            try:
                batch = self.client.batches.create(model=self.params.model_name, src=requests)
                logger.info(f"  📤 Batch {i+1}/{n_batches} | ID: {batch.name} | {len(requests)} requests")
                job_state["batches"].append({"batch_name": batch.name, "status": "SUBMITTED", "docs": chunk})
                time.sleep(0.3)
            except Exception as exc:
                logger.error(f"  ❌ Batch {i+1} submit fail: {exc} → direct queue")
                job_state["direct_docs"].extend(chunk)

        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(job_state, f, ensure_ascii=False, indent=2)

        logger.info(
            f"[submit] ✅ job_id={job_id} | "
            f"{len(job_state['batches'])} batches, {len(job_state['direct_docs'])} direct docs | "
            f"state → {state_file}"
        )
        return state_file

    def collect_batch_jobs(self, state_file: str) -> List[dict]:
        if not os.path.exists(state_file):
            raise FileNotFoundError(f"State file không tồn tại: {state_file}")

        with open(state_file, "r", encoding="utf-8") as f:
            job_state = json.load(f)

        job_id = job_state.get("job_id", "unknown")
        logger.info(
            f"[collect] job_id={job_id} | "
            f"{len(job_state['batches'])} batches | "
            f"{len(job_state['direct_docs'])} direct docs"
        )

        all_results: List[dict] = []

        direct_docs = job_state.get("direct_docs", [])
        if direct_docs:
            logger.info(f"[collect] 🔄 Direct fallback {len(direct_docs)} docs (từ submit fail)")
            for doc in tqdm(direct_docs, desc="direct-fallback", unit="doc"):
                all_results.append(self._extract_single_dict(doc))

        for batch_entry in tqdm(job_state["batches"], desc="Batches", unit="batch"):
            all_results.extend(self._process_one_batch(batch_entry["batch_name"], batch_entry["docs"]))

        results_file = state_file.replace(".json", ".results.json")
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        logger.info(f"[collect] ✅ job_id={job_id} | {len(all_results)} kết quả → {results_file}")
        return all_results

    def _make_batch_request(self, doc: dict) -> Optional[types.InlinedRequest]:
        doc_id, raw_text, hint = _parse_input_doc(doc)
        if not raw_text.strip():
            logger.warning(f"[{doc_id}] bỏ qua — văn bản rỗng")
            return None
        _, schema_class = _resolve_schema(doc_id, raw_text, hint)
        prompt = Prompt.build(
            schema_class, raw_text,
            max_chars=self.params.max_input_chars,
            max_chars_retry=self.params.max_input_chars_retry,
        )
        return types.InlinedRequest(
            contents=str(prompt),
            metadata={"doc_id": doc_id},
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                max_output_tokens=self.params.max_output_tokens,
                temperature=self.params.temperature,
            ),
        )

    def _poll_batch(self, batch_name: str) -> Any:
        deadline = time.time() + self.params.batch_timeout_sec
        while time.time() < deadline:
            batch = self.client.batches.get(name=batch_name)
            state = str(batch.state)
            if "SUCCEEDED" in state:
                return batch
            if "FAILED" in state or "CANCELLED" in state:
                logger.error(f"Batch {batch_name} → {state}")
                return batch
            logger.info(f"  ⏳ {batch_name} | {state} | chờ {self.params.batch_poll_interval}s...")
            time.sleep(self.params.batch_poll_interval)
        logger.error(f"Batch {batch_name} timeout sau {self.params.batch_timeout_sec}s")
        return self.client.batches.get(name=batch_name)

    def _collect_batch_results(
        self, batch: types.BatchJob, chunk: List[dict]
    ) -> Tuple[List[dict], List[str]]:
        results: List[dict] = []
        failed_ids: List[str] = []

        inlined = (batch.dest and batch.dest.inlined_responses) or []
        if len(inlined) != len(chunk):
            logger.warning(
                f"Số responses ({len(inlined)}) ≠ số requests ({len(chunk)}) — "
                f"một số doc có thể bị thiếu"
            )

        for doc, resp in zip(chunk, inlined):
            doc_id, raw_text, hint = _parse_input_doc(doc)
            schema_name, schema_class = _resolve_schema(doc_id, raw_text, hint)
            if resp.error:
                logger.error(f"[{doc_id}] batch response error: {resp.error}")
                failed_ids.append(doc_id)
                continue
            try:
                raw = ""
                if resp.response and resp.response.candidates:
                    candidate = resp.response.candidates[0]
                    parts = getattr(candidate.content, "parts", None) if candidate.content else None
                    if parts:
                        part = parts[0]
                        text_value = getattr(part, "text", "")
                        if isinstance(text_value, str):
                            raw = text_value
                        elif isinstance(text_value, list):
                            raw = "".join(str(item) for item in text_value)
                        else:
                            raw = str(text_value)
                result = _build_result(
                    doc_id, schema_name, schema_class, raw,
                    self.params.low_confidence_threshold,
                )
                results.append(result.to_dict(document_id=doc_id))
            except Exception as exc:
                logger.error(f"[{doc_id}] lỗi đọc response: {exc}")
                failed_ids.append(doc_id)

        return results, failed_ids

    def _process_one_batch(self, batch_name: Optional[str], chunk: List[dict]) -> List[dict]:
        if batch_name is None:
            logger.info(f"  🔄 Direct fallback cho {len(chunk)} doc (batch_name=None)")
            return [self._extract_single_dict(doc) for doc in tqdm(chunk, leave=False)]

        batch = self._poll_batch(batch_name)
        state = str(batch.state)

        if "SUCCEEDED" not in state:
            logger.warning(f"  ⚠️  Batch {batch_name} {state} → direct fallback {len(chunk)} doc")
            return [self._extract_single_dict(doc) for doc in tqdm(chunk, desc="direct fallback", leave=False)]

        results, failed_ids = self._collect_batch_results(batch, chunk)

        returned_ids = {r["document_id"] for r in results}
        doc_by_id = {_parse_input_doc(d)[0]: d for d in chunk}
        missing = [
            doc_by_id[fid] for fid in failed_ids if fid in doc_by_id
        ] + [
            d for d in chunk if _parse_input_doc(d)[0] not in returned_ids
        ]
        if missing:
            logger.warning(f"  ⚠️  {len(missing)} doc thiếu → direct retry")
            for doc in missing:
                results.append(self._extract_single_dict(doc))

        logger.info(f"  ✅ Batch {batch_name} → {len(results)} kết quả")
        return results

    def _extract_single_dict(self, doc: dict) -> dict:
        doc_id, raw_text, hint = _parse_input_doc(doc)
        schema_name, schema_class = _resolve_schema(doc_id, raw_text, hint)
        last_raw = ""
        for attempt in range(1, self.params.max_retries + 2):
            try:
                prompt = Prompt.build(
                    schema_class, raw_text, retry=(attempt > 1),
                    max_chars=self.params.max_input_chars,
                    max_chars_retry=self.params.max_input_chars_retry,
                )
                t0 = time.time()
                resp = self.client.models.generate_content(
                    model=self.params.model_name,
                    contents=str(prompt),
                    config=self._gen_cfg,
                )
                last_raw = resp.text or ""
                elapsed = time.time() - t0
                if clean_and_parse(last_raw, doc_id):
                    logger.info(f"[{doc_id}] ✅ direct OK lần {attempt} ({elapsed:.1f}s)")
                    return _build_result(
                        doc_id, schema_name, schema_class, last_raw,
                        self.params.low_confidence_threshold,
                    ).to_dict(document_id=doc_id)
                logger.warning(f"[{doc_id}] ⚠️  parse fail lần {attempt}")
            except Exception as exc:
                logger.error(f"[{doc_id}] ❌ lần {attempt}: {exc}")
                time.sleep(self.params.retry_delay * attempt)

        return {
            "document_id":        doc_id,
            "classification":     schema_name,
            "confidence_overall": 0.0,
            "fill_rate":          0.0,
            "error":              "failed_after_retries",
            "metadata":           [],
            "raw_output_sample":  last_raw[:300],
        }

    @staticmethod
    def _dict_to_result(d: dict) -> ExtractorResult:
        return ExtractorResult(
            schema_used=d.get("classification", ""),
            extracted_dynamic_data={m["field_name"]: m for m in d.get("metadata", [])} or None,
            confidence_overall=d.get("confidence_overall", 0.0),
            fill_rate=d.get("fill_rate", 0.0),
            error=d.get("error"),
        )