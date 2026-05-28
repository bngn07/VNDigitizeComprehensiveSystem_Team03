from __future__ import annotations

LABEL_SKIP = 1
LABEL_HEAVY = 2
LABEL_CLEAN = 3

LABEL_NAMES: dict[int, str] = {
    LABEL_SKIP: "SKIP",
    LABEL_HEAVY: "HEAVY",
    LABEL_CLEAN: "CLEAN",
}

FEATURE_KEYS: list[str] = [
    "white_ratio",
    "std_val",
    "coeff_variation",
    "laplacian_var",
    "mean_intensity",
    "edge_density",
]

FALLBACK_RULES: dict[str, float] = {
    "skip_white_ratio_gt": 0.95,
    "skip_std_lt": 15.0,
    "clean_laplacian_var_lt": 500.0,
    "clean_coeff_lt": 0.12,
}

RECIPES: dict[int, list[str]] = {
    LABEL_SKIP: [],
    LABEL_CLEAN: ["grayscale", "deskew", "autocrop", "qr_detect"],
    LABEL_HEAVY: [
        "grayscale",
        "denoise",
        "adaptive_threshold",
        "deskew",
        "autocrop",
        "qr_detect",
        "sharpen",
    ],
}

RESIZE_WIDTH = 500
RESIZE_HEIGHT = 500
WHITE_RATIO_TOLERANCE = 20
COEFF_EPSILON = 1e-6
