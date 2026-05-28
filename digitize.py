from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np

from src.preprocessing.preprocess import PreprocessResult, Preprocessing

from src.ocr.base import BaseOCR
from src.ocr.adapter import OCRAdapter
from src.ocr.confidence import ConfidenceScorer

from src.utils.extracting.correction.autocorrect import AutoCorrector


@dataclass
class DigitizeResult:
    # Images
    original_image: np.ndarray
    preprocessed_image: np.ndarray

    # Preprocessing
    preprocess: PreprocessResult

    # OCR (high‑level, with words and confidence if adapter + scorer are used)
    ocr: Any

    # Post‑OCR correction
    corrected_texts: Optional[List[str]] = None

    # LLM extraction (temporarily disabled)
    extraction: Optional[Any] = None


@dataclass
class DigitizeConfig:
    preprocessing: Dict[str, Any] | None = None
    autocorrect: Dict[str, Any] | None = None
    confidence: Dict[str, Any] | None = None


def _build_ocr_engine(ocr: BaseOCR) -> BaseOCR:
    return ocr


def _build_autocorrect(config: Dict[str, Any] | None) -> AutoCorrector:
    return AutoCorrector(config or {})


def _build_confidence_scorer(config: Dict[str, Any] | None) -> Optional[ConfidenceScorer]:
    cfg = config or {}
    if not cfg.get("enabled", False):
        return None

    threshold = float(cfg.get("threshold", 0.8))
    return ConfidenceScorer(threshold=threshold)


class Digitize:
    def __init__(
        self,
        ocr: BaseOCR,
        config: Dict[str, Any] | None = None,
    ) -> None:
        cfg = DigitizeConfig(**(config or {}))

        self.preprocess = Preprocessing(cfg.preprocessing or {})
        self.ocr_engine = _build_ocr_engine(ocr)
        self.autocorrect = _build_autocorrect(cfg.autocorrect)
        self.confidence_scorer = _build_confidence_scorer(cfg.confidence)
        self.adapter = OCRAdapter()

    def _run_preprocess(self, image: np.ndarray) -> PreprocessResult:
        return self.preprocess.process(image)

    def _run_ocr(self, image: np.ndarray) -> Any:
        low_level = self.ocr_engine.recognize(image)
        high_level = self.adapter.convert(low_level)

        if self.confidence_scorer is not None:
            high_level = self.confidence_scorer.score(high_level)

        return high_level

    def _run_correction(self, ocr_result: Any) -> List[str]:
        correction_result = self.autocorrect.correct_ocr_result(ocr_result)
        return correction_result.corrected_texts

    def _digitize_single(self, image: np.ndarray) -> DigitizeResult:
        pre = self._run_preprocess(image)
        ocr_result = self._run_ocr(pre.image)

        corrected_texts: List[str] | None = None
        if getattr(self.autocorrect, "enabled", False):
            corrected_texts = self._run_correction(ocr_result)

        extraction_result: Optional[Any] = None  # extraction disabled for now

        return DigitizeResult(
            original_image=image,
            preprocessed_image=pre.image,
            preprocess=pre,
            ocr=ocr_result,
            corrected_texts=corrected_texts,
            extraction=extraction_result,
        )

    def _digitize_batch(self, images: List[np.ndarray]) -> List[DigitizeResult]:
        pres: List[PreprocessResult] = [self._run_preprocess(img) for img in images]

        ocr_inputs = [p.image for p in pres]
        low_level_outputs = self.ocr_engine.recognize(ocr_inputs)
        if not isinstance(low_level_outputs, list):
            low_level_outputs = [low_level_outputs]

        high_level_outputs: List[Any] = []
        for low in low_level_outputs:
            high = self.adapter.convert(low)
            if self.confidence_scorer is not None:
                high = self.confidence_scorer.score(high)
            high_level_outputs.append(high)

        results: List[DigitizeResult] = []
        for original, pre, ocr_result in zip(images, pres, high_level_outputs):
            corrected_texts: List[str] | None = None
            if getattr(self.autocorrect, "enabled", False):
                corrected_texts = self._run_correction(ocr_result)

            extraction_result: Optional[Any] = None  # extraction disabled

            results.append(
                DigitizeResult(
                    original_image=original,
                    preprocessed_image=pre.image,
                    preprocess=pre,
                    ocr=ocr_result,
                    corrected_texts=corrected_texts,
                    extraction=extraction_result,
                )
            )

        return results

    def digitize(
        self,
        image: Union[np.ndarray, List[np.ndarray]],
    ) -> Union[DigitizeResult, List[DigitizeResult]]:
        if isinstance(image, list):
            return self._digitize_batch(image)
        return self._digitize_single(image)