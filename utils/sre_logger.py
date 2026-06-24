#!/usr/bin/env python3
"""
sre_logger.py — SRE-Observable Logging Middleware for Multimodal Sentiment Pipeline
===============================================================================================
Author:  Ridhi Jain  |  Date: 24 June 2026
Project: Multimodal-Sentiment-Analysis-AI (IBM SkillsBuild Track)

Provides structured JSON logging for every stage of the sentiment classification
pipeline, capturing execution latency, model version, token consumption, and
exception details. Implements the Failure Mode and Effects Analysis (FMEA) logging
schema for production-grade observability.
"""
import json
import logging
import time
import functools
import sys
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional


class SREJsonFormatter(logging.Formatter):
    """Outputs log records as single-line JSON strings for log aggregation."""
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if hasattr(record, "sre_metadata"):
            log_entry["sre"] = record.sre_metadata
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = {
                "class": type(record.exc_info[1]).__name__,
                "message": str(record.exc_info[1]),
            }
        return json.dumps(log_entry, default=str)


FMEA_FAILURE_MODES = {
    "FAIL_MODEL_LOAD": {
        "code": "FMEA-001",
        "description": "Transformers model failed to load from HuggingFace Hub",
        "severity": "critical",
        "recovery": "Retry 3× with exponential backoff; fall back to cached local model",
    },
    "FAIL_GEMINI_API": {
        "code": "FMEA-002",
        "description": "Gemini API returned non-200 or timed out during strategy generation",
        "severity": "high",
        "recovery": "Retry 2× (1s → )s); return cached last-known good response",
    },
    "FAIL_SPEECH_RECOGNITION": {
        "code": "FMEA-003",
        "description": "SpeechRecognition engine failed to transcribe audio (noise, format, quota)",
        "severity": "medium",
        "recovery": "Fall back to Google Speech API; if both fail, return empty transcript",
    },
    "FAIL_VIDEO_FRAME_EXTRACTION": {
        "code": "FMEA-004",
        "description": "OpenCV failed to extract frame from uploaded video (corrupted file, codec)",
        "severity": "medium",
        "recovery": "Skip video modality; process remaining text/audio/CSV",
    },
    "FAIL_CSV_PARSE": {
        "code": "FMEA-005",
        "description": "Uploaded CSV file could not be parsed by pandas (encoding, format)",
        "severity": "medium",
        "recovery": "Attempt UTF-8 → Latin-1 → cp1252 fallback encoding chain",
    },
    "FAIL_SENTIMENT_CLASSIFICATION": {
        "code": "FMEA-006",
        "description": "DistilBERT sentiment pipeline returned unexpected output or NaN score",
        "severity": "high",
        "recovery": "Return neutral sentiment with confidence=0.0; log anomaly for review",
    },
    "MISSING_API_KEY": {
        "code": "FMEA-007",
        "description": "GEMINI_API_KEY not found in st.secrets or environment",
        "severity": "critical",
        "recovery": "Disable GenAI Strategy tab; multimodal analysis tabs remain active",
    },
}


class SRELogger:
    """
    Structured SRE logger for the multimodal sentiment pipeline.
    Captures latency, model metadata, exceptions, and FMEA failure codes.
    """

    def __init__(self, name: str, level: int = logging.INFO):
        self.logger = logging.getLogger(f"sre.{name}")
        self.logger.setLevel(level)
        self.logger.propagate = False
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(SREJsonFormatter())
            self.logger.addHandler(handler)
        self._failure_counts: Dict[str, int] = {}

    def log_latency(self, operation: str, latency_ms: float, metadata: Optional[Dict[str, Any]] = None) -> None:
        extra = {
            "sre_metadata": {
                "operation": operation,
                "latency_ms": round(latency_ms, 2),
                "sla_ms": self._get_sla(operation),
                "sla_violated": latency_ms > self._get_sla(operation),
                **(metadata or {}),
            }
        }
        self.logger.info(f"{operation}: {latency_ms:.2f}ms", extra=extra)

    def log_failure(self, failure_key: str, detail: Optional[str] = None, exc: Optional[Exception] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        mode = FMEA_FAILURE_MODES.get(failure_key, {
            "code": "FMEA-UNKNOWN",
            "description": "Uncategorized failure",
            "severity": "low",
            "recovery": "No recovery defined",
        })
        self._failure_counts[failure_key] = self._failure_counts.get(failure_key, 0) + 1
        extra = {
            "sre_metadata": {
                "fmea_code": mode["code"],
                "severity": mode["severity"],
                "failure_key": failure_key,
                "description": mode["description"],
                "recovery_action": mode["recovery"],
                "failure_count": self._failure_counts[failure_key],
                "detail": detail,
                "exception_class": type(exc).__name__ if exc else None,
                "exception_message": str(exc) if exc else None,
                **(metadata or {}),
            }
        }
        self.logger.warning(f"[{mode['code']}] {mode['description']}: {detail or 'no detail'}", extra=extra)

    def instrument(self, operation: str, metadata: Optional[Dict[str, Any]] = None):
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    elapsed = (time.perf_counter() - start) * 1000
                    self.log_latency(operation, elapsed, {
                        **(metadata or {}),
                        "function": func.__name__,
                        "status": "success",
                    })
                    return result
                except Exception as e:
                    elapsed = (time.perf_counter() - start) * 1000
                    fmea_key = self._map_exception_to_fmea(e, operation)
                    self.log_failure(fmea_key, detail=str(e), exc=e, metadata={
                        "operation": operation,
                        "latency_ms": round(elapsed, 2),
                        **(metadata or {}),
                    })
                    raise
            return wrapper
        return decorator

    def telemetry_summary(self) -> Dict[str, Any]:
        return {
            "total_failure_events": sum(self._failure_counts.values()),
            "failure_breakdown": dict(self._failure_counts),
            "fmea_codes": list(FMEA_FAILURE_MODES.keys()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def _get_sla(operation: str) -> float:
        slas = {
            "text_sentiment": 200.0,
            "image_emotion": 500.0,
            "speech_recognition": 2000.0,
            "csv_parse": 300.0,
            "genai_strategy": 5000.0,
            "model_load": 10000.0,
        }
        return slas.get(operation, 1000.0)

    @staticmethod
    def _map_exception_to_fmea(exc: Exception, operation: str) -> str:
        exc_name = type(exc).__name__
        mapping = {
            "OSError": "FAIL_MODEL_LOAD",
            "ValueError": "FAIL_SENTIMENT_CLASSIFICATION",
            "RequestException": "FAIL_GEMINI_API",
            "RequestError": "FAIL_GEMINI_API",
            "UnknownValueError": "FAIL_SPEECH_RECOGNITION",
            "ParserError": "FAIL_CSV_PARSE",
            "KeyError": "FAIL_MODEL_LOAD",
        }
        return mapping.get(exc_name, f"FAIL_%{operation.upper()}")


_sre_logger_instance: Optional[SRELogger] = None

def get_sre_logger() -> SRELogger:
    global _sre_logger_instance
    if _sre_logger_instance is None:
        _sre_logger_instance = SRELogger("multimodal-sentiment")
    return _sre_logger_instance
