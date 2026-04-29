client ──► DocumentReader (Facade)
              │
              ▼
          OCRPipeline ──► [Preprocess] → [Detect] → [Recognize] → [Postprocess]
                              │            │            │              │
                              ▼            ▼            ▼              ▼
                          IPreprocessor IDetector  IRecognizer   IPostprocessor
                          (Strategy)   (Strategy)   (Strategy)    (Strategy)
                              ▲            ▲            ▲              ▲
                              └──────── StrategyFactory(config) ───────┘

ocr_system/
├── core/
│   ├── __init__.py
│   ├── data_models.py
│   └── stage.py
├── strategies/
│   ├── __init__.py
│   ├── base.py
│   ├── preprocessor.py
│   ├── detector.py
│   ├── recognizer.py
│   └── postprocessor.py
├── stages/
│   ├── __init__.py
│   └── ocr_stages.py
├── pipeline.py
├── factory.py
├── reader.py
└── main.py