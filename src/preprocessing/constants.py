from __future__ import annotations

from enum import IntEnum, StrEnum

class Label(IntEnum):
    SKIP  = 1
    HEAVY = 2
    CLEAN = 3

class PreprocessError(StrEnum):
    EMPTY_IMAGE          = "Input image is empty."
    CORRUPTED_IMAGE      = "Corrupted image data."
    FILE_TYPE_UNKNOWN    = "Could not determine file type: {detail}"
    FILE_TYPE_PDF        = "PDF file detected, expected an image."
    FILE_TYPE_VIDEO      = "Video file detected ({detail}), expected an image."
    FILE_TYPE_UNSUPPORTED = "Unsupported file type: {detail}"

    def format(self, **kwargs) -> str:
        return self.value.format(**kwargs)

class Step(StrEnum):
    GRAYSCALE           = "grayscale"
    ORIENTATION         = "orientation"
    PERSPECTIVE         = "perspective"
    DENOISE             = "denoise"
    DESKEW              = "deskew"
    AUTOCROP            = "autocrop"
    ADAPTIVE_THRESHOLD  = "adaptive_threshold"
    SHARPEN             = "sharpen"
    ENHANCE_CONTRAST    = "enhance_contrast"
    LEVELS              = "levels"
    QR_DETECT           = "qr_detect"

LABEL_SKIP  = Label.SKIP
LABEL_HEAVY = Label.HEAVY
LABEL_CLEAN = Label.CLEAN

LABEL_NAMES: dict[int, str] = {label: label.name for label in Label}

FEATURE_KEYS: list[str] = [
    "white_ratio", "std_val", "coeff_variation", 
    "laplacian_var", "mean_intensity", "edge_density"
]

FALLBACK_RULES: dict[str, float] = {
    "skip_white_ratio_gt":    0.95,
    "skip_std_lt":            15.0,
    "clean_laplacian_var_lt": 500.0,
    "clean_coeff_lt":         0.12,
}

RECIPES: dict[int, list[Step]] = {
    Label.SKIP:  [],
    Label.CLEAN: [Step.GRAYSCALE, Step.ORIENTATION, Step.DENOISE, Step.LEVELS, Step.DESKEW, Step.AUTOCROP],
    Label.HEAVY: [Step.GRAYSCALE, Step.ORIENTATION, Step.DENOISE, Step.LEVELS, Step.ADAPTIVE_THRESHOLD, Step.DESKEW, Step.AUTOCROP],
}

RESIZE_WIDTH = 500
RESIZE_HEIGHT = 500
WHITE_RATIO_TOLERANCE = 20
COEFF_EPSILON = 1e-6

STEP_PARAMS = {
    Label.CLEAN: {
        Step.DENOISE: {"kernel": 3},
        Step.LEVELS:  {"black": 20, "white": 180},
    },
    Label.HEAVY: {
        Step.DENOISE:            {"kernel": 5},
        Step.LEVELS:             {"black": 20, "white": 180},
        Step.ADAPTIVE_THRESHOLD: {"block_size": 21, "C": 8},
    },
}