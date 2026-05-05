vn-digitize/
│
├── README.md
├── DEV_LOG.md
├── requirements.txt
├── .gitignore
│
├── src/
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── blank.py
│   │   ├── code.py
│   │   └── preprocess.py
│   │
│   ├── ocr/
│   │   ├── __init__.py
│   │   ├── engine.py
│   │   └── schema.py
│   │
│   ├── extraction/
│   │   ├── __init__.py
│   │   ├── rules.py
│   │   └── extractor.py
│   │
│   ├── pipeline.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   └── models.py
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── image_io.py
│   │   └── logger.py
│   │
│   └── config.py
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   │
│   ├── unit/
│   │   ├── test_blank.py
│   │   ├── test_orientation.py
│   │   ├── test_barcode.py
│   │   ├── test_extraction.py
│   │   └── test_pipeline.py
│   │
│   └── integration/
│       └── test_api.py
│
├── data/
│   ├── test_images/
│   │   ├── blank/
│   │   ├── rotated/
│   │   ├── with_text/
│   │   └── with_barcode/
│   │
│   └── models/
│       ├── blank_classifier.pkl
│       └── .gitkeep
│
├── notebooks/
│   ├── 01_explore_ocr_libs.ipynb
│   ├── 02_train_blank_classifier.ipynb
│   └── 03_benchmark_accuracy.ipynb
│   
├── scripts/
│   ├── benchmark.py
│   └── eval_accuracy.py
│
└── docs/
    ├── architecture.md
    ├── api_contract.md
    └── decisions.md