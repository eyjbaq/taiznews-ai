"""Quota and Rate Limiting Manager for Google GenAI models with 24-hour reset tracking."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

LOGGER = logging.getLogger(__name__)

DEFAULT_USAGE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "gemini_usage.json"

# Models and their official operational limits
MODEL_SPECS: Dict[str, dict] = {
    "gemini-3.6-flash": {
        "daily_limit": 20,
        "max_rpm": 3,
        "min_interval": 20.0,
        "is_lite": False,
    },
    "gemini-3.5-flash": {
        "daily_limit": 20,
        "max_rpm": 3,
        "min_interval": 20.0,
        "is_lite": False,
    },
    "gemini-2.5-flash": {
        "daily_limit": 20,
        "max_rpm": 3,
        "min_interval": 20.0,
        "is_lite": False,
    },
    "gemini-3.5-flash-lite": {
        "daily_limit": 500,
        "max_rpm": 10,
        "min_interval": 6.0,
        "is_lite": True,
    },
    "gemini-3.1-flash-lite": {
        "daily_limit": 500,
        "max_rpm": 10,
        "min_interval": 6.0,
        "is_lite": True,
    },
}

DEFAULT_MODEL_ORDER: List[str] = list(MODEL_SPECS.keys())


class RateLimiter:
    """Enforces per-model rate limits (Standard: 3 req/min; Lite: 10 req/min)."""

    def __init__(self) -> None:
        self._timestamps: Dict[str, deque[float]] = {}
        self._last_request_time: Dict[str, float] = {}
        self._lock = threading.Lock()

    def get_limits_for_model(self, model_name: str) -> Tuple[int, float]:
        """Return (max_rpm, min_interval_seconds) for the specified model."""
        spec = MODEL_SPECS.get(model_name)
        if spec:
            return spec["max_rpm"], spec["min_interval"]
        # Fallback heuristic: check if name contains 'lite'
        if "lite" in model_name.lower():
            return 10, 6.0
        return 3, 20.0

    def wait_for_slot(self, model_name: str) -> float:
        """Block until a request slot is legally available without violating rate limits."""
        max_rpm, min_interval = self.get_limits_for_model(model_name)

        with self._lock:
            now = time.time()
            ts = self._timestamps.setdefault(model_name, deque())

            # Clean timestamps older than 60 seconds
            while ts and (now - ts[0]) >= 60.0:
                ts.popleft()

            wait_time = 0.0

            # Condition 1: Enforce minimum interval between consecutive requests to this model
            last = self._last_request_time.get(model_name, 0.0)
            if last > 0:
                elapsed = now - last
                if elapsed < min_interval:
                    wait_time = max(wait_time, min_interval - elapsed)

            # Condition 2: Enforce max requests in rolling 60-second window
            if len(ts) >= max_rpm:
                oldest = ts[0]
                window_wait = 60.0 - (now - oldest) + 0.5
                wait_time = max(wait_time, window_wait)

            if wait_time > 0.2:
                print(
                    f"⏳ [معدل الطلبات] انتظار {wait_time:.1f} ثانية للنموذج [{model_name}] (الحد: {max_rpm} طلبات/دقيقة)..."
                )
                time.sleep(wait_time)

            now = time.time()
            ts.append(now)
            self._last_request_time[model_name] = now
            return wait_time


class QuotaManager:
    """Tracks model usage, 24-hour quota reset windows, and automatic cascading."""

    def __init__(self, storage_path: Path = DEFAULT_USAGE_PATH) -> None:
        self.storage_path = storage_path
        self._lock = threading.Lock()
        self._ensure_storage()

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    def _ensure_storage(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.storage_path.exists():
            initial_data: Dict[str, dict] = {}
            for model, spec in MODEL_SPECS.items():
                initial_data[model] = {
                    "usage_count": 0,
                    "daily_limit": spec["daily_limit"],
                    "first_request_at": None,
                    "exhausted_at": None,
                    "exhausted_reason": None,
                }
            self._save_raw(initial_data)

    def _load_data(self) -> Dict[str, dict]:
        now = self._now()
        data: Dict[str, dict] = {}
        try:
            if self.storage_path.exists():
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                    # Support legacy format if present
                    if "usage" in raw and "exhausted_models" in raw:
                        legacy_usage = raw.get("usage", {})
                        legacy_exhausted = raw.get("exhausted_models", [])
                        for model, spec in MODEL_SPECS.items():
                            cnt = legacy_usage.get(model, 0)
                            is_ex = model in legacy_exhausted
                            data[model] = {
                                "usage_count": cnt,
                                "daily_limit": spec["daily_limit"],
                                "first_request_at": None,
                                "exhausted_at": now.isoformat() if is_ex else None,
                                "exhausted_reason": "429 Quota Exceeded" if is_ex else None,
                            }
                    else:
                        data = raw
        except Exception as err:
            LOGGER.warning("Could not parse quota file, reinitializing: %s", err)

        modified = False
        # Ensure all defined models exist in data and evaluate 24-hour resets
        for model, spec in MODEL_SPECS.items():
            entry = data.setdefault(model, {
                "usage_count": 0,
                "daily_limit": spec["daily_limit"],
                "first_request_at": None,
                "exhausted_at": None,
                "exhausted_reason": None,
            })

            # Check 24-hour reset for exhausted models
            exhausted_at_str = entry.get("exhausted_at")
            if exhausted_at_str:
                try:
                    exhausted_dt = datetime.fromisoformat(exhausted_at_str)
                    if (now - exhausted_dt) >= timedelta(hours=24):
                        # 24 hours have passed! Automatically reset model
                        entry["usage_count"] = 0
                        entry["first_request_at"] = None
                        entry["exhausted_at"] = None
                        entry["exhausted_reason"] = None
                        modified = True
                        print(
                            f"🔄 [تجديد الحصة] مرت 24 ساعة على استنفاد النموذج [{model}]. تم تصفير العداد وأصبح متاحاً للاستخدام!"
                        )
                except Exception as parse_err:
                    LOGGER.warning("Failed parsing exhausted_at for %s: %s", model, parse_err)

            # Check 24-hour rolling reset for usage count
            first_req_str = entry.get("first_request_at")
            if first_req_str and not entry.get("exhausted_at"):
                try:
                    first_dt = datetime.fromisoformat(first_req_str)
                    if (now - first_dt) >= timedelta(hours=24):
                        entry["usage_count"] = 0
                        entry["first_request_at"] = None
                        modified = True
                except Exception:
                    pass

        if modified or not self.storage_path.exists():
            self._save_raw(data)

        return data

    def _save_raw(self, data: dict) -> None:
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            temp_file = self.storage_path.with_suffix(".tmp")
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            temp_file.replace(self.storage_path)
        except Exception as err:
            LOGGER.error("Failed to persist quota data: %s", err)

    def get_candidate_models(self) -> List[str]:
        """Return the prioritized list of supported models, honoring GEMINI_MODEL env if set."""
        preferred_env = os.getenv("GEMINI_MODEL")
        chain = []
        if preferred_env and preferred_env.strip() in MODEL_SPECS:
            chain.append(preferred_env.strip())

        for model in DEFAULT_MODEL_ORDER:
            if model not in chain:
                chain.append(model)
        return chain

    def get_available_models(self) -> List[str]:
        """Return models that have not reached their daily limit and are not exhausted."""
        with self._lock:
            data = self._load_data()
            available = []
            for model in self.get_candidate_models():
                entry = data.get(model, {})
                usage = entry.get("usage_count", 0)
                limit = entry.get("daily_limit", MODEL_SPECS.get(model, {}).get("daily_limit", 20))
                exhausted_at = entry.get("exhausted_at")

                if not exhausted_at and usage < limit:
                    available.append(model)
            return available

    def record_success(self, model_name: str) -> int:
        """Increment usage counter upon a successful API call and record timestamps."""
        with self._lock:
            data = self._load_data()
            now_iso = self._now().isoformat()
            entry = data.setdefault(model_name, {
                "usage_count": 0,
                "daily_limit": MODEL_SPECS.get(model_name, {}).get("daily_limit", 20),
                "first_request_at": None,
                "exhausted_at": None,
                "exhausted_reason": None,
            })

            if not entry.get("first_request_at"):
                entry["first_request_at"] = now_iso

            entry["last_request_at"] = now_iso
            current_count = entry.get("usage_count", 0) + 1
            entry["usage_count"] = current_count

            limit = entry.get("daily_limit", 20)
            if current_count >= limit:
                entry["exhausted_at"] = now_iso
                entry["exhausted_reason"] = f"Reached daily limit ({current_count}/{limit})"
                print(
                    f"⚠️ وصل النموذج [{model_name}] إلى حده الأقصى ({current_count}/{limit} طلب). سيتجدد بعد 24 ساعة."
                )

            self._save_raw(data)
            return current_count

    def mark_exhausted(self, model_name: str, reason: str = "") -> None:
        """Mark a model as exhausted for 24 hours with exact timestamp and reason."""
        with self._lock:
            data = self._load_data()
            now_iso = self._now().isoformat()
            entry = data.setdefault(model_name, {
                "usage_count": 0,
                "daily_limit": MODEL_SPECS.get(model_name, {}).get("daily_limit", 20),
                "first_request_at": None,
                "exhausted_at": None,
                "exhausted_reason": None,
            })
            entry["exhausted_at"] = now_iso
            entry["exhausted_reason"] = reason or "Quota Exceeded"
            self._save_raw(data)
            LOGGER.info("Model %s marked exhausted at %s: %s", model_name, now_iso, reason)

    def get_status_summary(self) -> str:
        """Return formatted status summary showing usage, limits, and countdown timers."""
        with self._lock:
            data = self._load_data()
            now = self._now()
            parts = []

            for model in self.get_candidate_models():
                entry = data.get(model, {})
                usage = entry.get("usage_count", 0)
                limit = entry.get("daily_limit", MODEL_SPECS.get(model, {}).get("daily_limit", 20))
                exhausted_at_str = entry.get("exhausted_at")
                reason = entry.get("exhausted_reason", "")

                if "404" in str(reason):
                    parts.append(f"{model}: [غير متاح 404 ❌]")
                    continue

                if exhausted_at_str:
                    try:
                        ex_dt = datetime.fromisoformat(exhausted_at_str)
                        rem_seconds = max(0.0, (timedelta(hours=24) - (now - ex_dt)).total_seconds())
                        rem_hours = rem_seconds / 3600.0
                        parts.append(f"{model}: {usage}/{limit} [مستنفد ⛔ - يتاح بعد {rem_hours:.1f} ساعة]")
                    except Exception:
                        parts.append(f"{model}: {usage}/{limit} [مستنفد ⛔]")
                elif usage >= limit:
                    parts.append(f"{model}: {usage}/{limit} [مكتمل الحد 🛑]")
                else:
                    parts.append(f"{model}: {usage}/{limit} [متاح ✅]")

            return " | ".join(parts)


# Global singletons
RATE_LIMITER = RateLimiter()
QUOTA_MANAGER = QuotaManager()
