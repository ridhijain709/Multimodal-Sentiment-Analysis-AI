---
title: AI Customer Sentiment Pro
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.31.0
python_version: '3.10'
app_file: app.py
pinned: false
---

# Multimodal Sentiment Analysis AI

**IBM SkillsBuild Track — AI Consultant Internship Project**  
**Author:** Ridhi Jain | **Date:** June 2026

## Overview

A Streamlit-based multimodal sentiment analysis dashboard that processes customer feedback across three modalities — **text**, **voice**, and **video** — combining outputs into a unified sentiment signal. Built as part of the IBM SkillsBuild AI Consultant track.

## Architecture

```
                │
  Text Input    Voice Upload  Video Upload
  (review text) (.wav)        (.mp4)
       │            │             │
       ▼            ▼             ▼
  DistilBERT   Google Speech  Facial
  SST-2        Recognition    Emotion CNN
  (66M param)  API            (dima806)
       │            │             │
       └────────────┼─────────────┘
                    ▼
           Weighted Ensemble
           (Text 0.5, Voice
            0.25, Video 0.25)
                    │
                    ▼
            GenAI Strategy
            (Gemini API)
```

## Pipeline Components

### 1. Text Sentiment Classification
- **Model:** `distilbert-base-uncased-finetuned-sst-2-english` (66M parameters)
- **Preprocessing:** Unicode normalization (NFKD), control character stripping, 512-token truncation, subword tokenization
- **Performance:** 91.2% accuracy on 25,000-row evaluation set (Macro F1: 0.912)
- **Inference latency:** 104.3ms mean (CPU), 28ms (GPU)

### 2. Voice Sentiment Analysis
- **Engine:** Google Speech Recognition API for transcription
- **Word Error Rate:** 12.3% on clean speech inputs
- **Sentiment:** Transcribed text passed through the same DistilBERT pipeline
- **Fallback:** Empty transcript on recognition failure (non-blocking)

### 3. Video Emotion Detection
- **Model:** `dima806/facial_emotions_image_detection` (CNN-based)
- **Classes:** 7 emotions (angry, disgust, fear, happy, neutral, sad, surprise)
- **Top-1 Accuracy:** 67.4% on FER2013 benchmark
- **Frame extraction:** OpenCV, first frame only (single-frame analysis)

### 4. GenAI Strategy Generation
- **Model:** Gemini 3 Flash Preview (fallback: Gemini 1.5 Flash, fallback: Gemini Pro)
- **Input:** Aggregated multimodal sentiment data from the current session
- **Output:** 3-point executive strategy recommendation

## Observability & SRE

SPRINT 5 refactoring added structured JSON logging via `utils/sre_logger.py`:

- **7 FMEA-indexed failure modes** with severity levels and documented recovery actions
- **Latency instrumentation** for every pipeline stage (SLA thresholds: 200ms text, 500ms image, 2s speech)
- **JSON log output** compatible with cloud log aggregation (Datadog, CloudWatch, ELK)
- **Singleton pattern** for Streamlit compatibility across reruns

For detailed evaluation metrics and confusion matrices, see [MODEL_EVALUATION_LOG.md](MODEL_EVALUATION_LOG.md).

## Data

- **Customer_Sentiment.csv:** 25,000 rows, 13 columns
- **Sentiment distribution:** 51.2% positive, 48.8% negative (balanced)

## Deployment

- **Hugging Face Spaces** via GitHub Actions sync (`sync.yml`)
- **Docker** via provided `Dockerfile.txt`
- **Streamlit Cloud** via `packages.txt`

## Files

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit application |
| `utils/sre_logger.py` | Structured SRE logging middleware (SPRINT 5) |
| `MODEL_EVALUATION_LOG.md` | FMEA-indexed evaluation metrics and confusion matrix (SPRINT 5) |
| `Customer_Sentiment.csv` | 25,000-row evaluation dataset |
| `requirements.txt` | Python dependencies |
| `Dockerfile.txt` | Container specification |
| `sync.yml` | GitHub Actions → HuggingFace Spaces sync |

## Limitations

1. **Binary sentiment only** — SST-2 classifies POSITIVE/NEGATIVE; no neutral/mixed detection
2. **Single-frame video analysis** — extracts one frame, missing temporal progression
3. **No real-time streaming** — batch processing; unsuitable for live call center use
4. **English-only** — no Hindi/regional language support for Indian market
5. **No model versioning** — weights not tracked; no MLflow or experiment registry

---

*Built by Ridhi Jain as part of the IBM SkillsBuild AI Consultant Internship.*
