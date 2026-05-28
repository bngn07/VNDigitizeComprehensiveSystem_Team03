import numpy as np
import pytest

from decision import DecisionResult, LABEL_CLEAN, LABEL_HEAVY, LABEL_SKIP
from preprocess import Preprocessing
import steps


@pytest.fixture
def color_image() -> np.ndarray:
    image = np.full((120, 160, 3), 255, dtype=np.uint8)
    image[30:90, 40:120] = (0, 0, 0)
    return image


@pytest.fixture
def gray_image() -> np.ndarray:
    image = np.full((120, 160), 255, dtype=np.uint8)
    image[30:90, 40:120] = 0
    return image


class StubEngine:
    def __init__(self, label: int, recipe: list[str], label_name: str = "TEST") -> None:
        self._result = DecisionResult(
            label=label,
            label_name=label_name,
            confidence=1.0,
            recipe=recipe,
            probs={"SKIP": 0.0, "HEAVY": 0.0, "CLEAN": 1.0},
        )

    def evaluate(self, image):
        return self._result


@pytest.fixture
def pipeline() -> Preprocessing:
    return Preprocessing(provider=None)


def test_process_empty_image_raises_by_default(pipeline: Preprocessing):
    with pytest.raises(ValueError):
        pipeline.process(None)


def test_tograyscale_converts_color_image(color_image: np.ndarray):
    gray = steps.to_grayscale(color_image)
    assert gray.ndim == 2
    assert gray.shape == color_image.shape[:2]


def test_tograyscale_returns_copy_for_gray_input(gray_image: np.ndarray):
    gray = steps.to_grayscale(gray_image)
    assert gray.ndim == 2
    assert np.array_equal(gray, gray_image)
    assert gray is not gray_image


def test_denoise_preserves_shape(gray_image: np.ndarray):
    denoised = steps.denoise(gray_image)
    assert denoised.shape == gray_image.shape


def test_deskew_preserves_shape(gray_image: np.ndarray):
    result = steps.deskew(gray_image)
    assert result.shape == gray_image.shape


def test_autocrop_returns_same_or_smaller_image(gray_image: np.ndarray):
    cropped = steps.autocrop(gray_image)
    assert cropped.shape[0] <= gray_image.shape[0]
    assert cropped.shape[1] <= gray_image.shape[1]


def test_adaptive_threshold_preserves_shape(gray_image: np.ndarray):
    thresholded = steps.adaptive_threshold(gray_image)
    assert thresholded.shape == gray_image.shape


def test_sharpen_preserves_shape(gray_image: np.ndarray):
    sharpened = steps.sharpen(gray_image)
    assert sharpened.shape == gray_image.shape


def test_applyrecipe_runs_known_steps(pipeline: Preprocessing, color_image: np.ndarray):
    # Using actual step names mapped in preprocess.STEP_MAP
    result = pipeline.apply_recipe(color_image, ["grayscale", "denoise", "deskew", "autocrop"])
    assert result.ndim == 2


def test_applyrecipe_unknown_step_raises_keyerror(pipeline: Preprocessing):
    image = np.zeros((50, 50), dtype=np.uint8)
    with pytest.raises(KeyError):
        pipeline.apply_recipe(image, ["unknownstep"])


def test_process_skip_recipe_returns_metadata(color_image: np.ndarray):
    pipeline = Preprocessing(provider=None)
    pipeline.engine = StubEngine(LABEL_SKIP, [], "SKIP")

    result = pipeline.process(color_image)

    assert result.image.shape == color_image.shape
    assert result.metadata["status"] == "skip"
    assert result.metadata["qrcodes"] == []
    assert result.decision.label == LABEL_SKIP


def test_process_clean_recipe_runs_pipeline(color_image: np.ndarray):
    pipeline = Preprocessing(provider=None)
    pipeline.engine = StubEngine(LABEL_CLEAN, ["grayscale", "deskew", "autocrop"], "CLEAN")

    result = pipeline.process(color_image)

    assert result.image.ndim == 2
    assert result.metadata["status"] == "clean"
    assert result.decision.label == LABEL_CLEAN


def test_process_heavy_recipe_runs_pipeline(color_image: np.ndarray):
    pipeline = Preprocessing(provider=None)
    pipeline.engine = StubEngine(
        LABEL_HEAVY,
        ["grayscale", "denoise", "adaptive_threshold", "deskew", "autocrop", "sharpen"],
        "HEAVY",
    )

    result = pipeline.process(color_image)

    assert result.image.ndim == 2
    assert result.metadata["status"] == "heavy"
    assert result.decision.label == LABEL_HEAVY


def test_perspectivecorrect_returns_original_when_document_not_found(
    color_image: np.ndarray,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(steps, "detect_document", lambda image: None)
    result = steps.perspective_correct(color_image)
    assert np.array_equal(result, color_image)