"""
Rate limit manager for Gemini API based on usage tiers
Implements RPM (Requests Per Minute), TPM (Tokens Per Minute), and RPD (Requests Per Day) limits
"""

import json
import logging
import threading
import time
from collections import deque
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import yaml

from .constants import REQUEST_WINDOW_SECONDS
from .misc import tz_now

logger = logging.getLogger(__name__)


class RateLimitManager:
    """Global rate limit manager for Gemini API calls"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return

        self._initialized = True
        self.tier = "free"
        self.model = "gemini-2.5-flash"

        # Configuration paths
        self.config_file = Path("settings") / "rate_limits.yaml"
        self.state_file = Path(".cache") / "rate_limit_state.json"
        self.state_file.parent.mkdir(exist_ok=True)

        # Load rate limits configuration
        self.rate_limits = self._load_rate_limits_config()
        self.default_model = self.rate_limits.get("default_model", "gemini-2.5-flash")

        # Request tracking - now model-specific
        self.model_states = {}  # Dict[model_name, dict] containing per-model tracking
        self.current_model = "gemini-2.5-flash"

        # Threading lock for thread safety
        self.request_lock = threading.Lock()

        # Load saved state
        self._load_state()

    def _get_model_state(self, model: str) -> dict:
        """Get or create state tracking for a specific model"""
        if model not in self.model_states:
            self.model_states[model] = {
                "request_times": deque(),
                "token_usage": deque(),
                "daily_requests": 0,
                "last_reset_date": tz_now().date(),
            }
        return self.model_states[model]

    def set_tier_and_model(self, tier: str, model: str):
        """Set the API tier and model for rate limiting"""
        if tier not in ["free", "tier1", "tier2", "tier3"]:
            raise ValueError(f"Invalid tier: {tier}. Must be one of: free, tier1, tier2, tier3")

        if model not in self.rate_limits.get("models", {}):
            logger.warning("Model %s not found in rate limits. Using default model %s.", model, self.default_model)
            model = self.default_model

        self.tier = tier
        self.model = model  # For compatibility
        self.current_model = model

        # Initialize model state if not exists
        self._get_model_state(model)

        self._save_state()
        logger.info("Rate limit manager set to %s tier with model %s", tier, model)

    def get_current_limits(self) -> dict[str, int | None]:
        """Get current rate limits for the configured tier and model"""
        models = self.rate_limits.get("models", {})
        if self.model not in models:
            logger.warning("Model %s not found, using default %s", self.model, self.default_model)
            model_config = models.get(self.default_model, {})
        else:
            model_config = models[self.model]

        return model_config.get(self.tier, {})

    def _load_rate_limits_config(self) -> dict[str, Any]:
        """Load rate limits configuration from YAML file"""
        if not self.config_file.exists():
            logger.error("Rate limits config file not found: %s", self.config_file)
            return self._get_fallback_config()

        try:
            with open(self.config_file, encoding="utf-8") as f:
                config = yaml.safe_load(f)

            logger.info("Loaded rate limits config from %s", self.config_file)
            return config

        except (OSError, yaml.YAMLError, UnicodeDecodeError) as e:
            logger.error("Failed to load rate limits config: %s", e)
            return self._get_fallback_config()

    def _get_fallback_config(self) -> dict[str, Any]:
        """Get fallback configuration when YAML file is not available"""
        logger.warning("Using fallback rate limits configuration")
        return {
            "models": {
                "gemini-2.5-flash": {
                    "free": {"rpm": 15, "tpm": 1500000, "rpd": 1500},
                    "tier1": {"rpm": 1000, "tpm": 4000000, "rpd": None},
                    "tier2": {"rpm": 1000, "tpm": 4000000, "rpd": None},
                    "tier3": {"rpm": 1000, "tpm": 4000000, "rpd": None},
                }
            },
            "default_model": "gemini-2.5-flash",
        }

    def _load_state(self):
        """Load rate limit state from file (supports both old and new format)"""
        if not self.state_file.exists():
            return

        try:
            with open(self.state_file) as f:
                data = json.load(f)

            # Load basic settings
            self.tier = data.get("tier", self.tier)
            self.current_model = data.get("current_model", data.get("model", self.current_model))
            self.model = self.current_model  # For compatibility

            now = time.time()

            # Check if this is new model-specific format
            if "models" in data:
                # New format: model-specific states
                for model_name, model_data in data["models"].items():
                    # Load dates
                    saved_date = model_data.get("last_reset_date")
                    if saved_date:
                        saved_date = date.fromisoformat(saved_date)
                        if saved_date != tz_now().date():
                            # Reset daily counter if date changed
                            daily_requests = 0
                            last_reset_date = tz_now().date()
                        else:
                            daily_requests = model_data.get("daily_requests", 0)
                            last_reset_date = saved_date
                    else:
                        daily_requests = model_data.get("daily_requests", 0)
                        last_reset_date = tz_now().date()

                    # Load request times (only keep recent ones)
                    request_times = model_data.get("request_times", [])
                    valid_request_times = deque(
                        t for t in request_times if now - t <= REQUEST_WINDOW_SECONDS
                    )

                    # Load token usage (only keep recent ones)
                    token_usage = model_data.get("token_usage", [])
                    valid_token_usage = deque(
                        (t, tokens) for t, tokens in token_usage if now - t <= REQUEST_WINDOW_SECONDS
                    )

                    self.model_states[model_name] = {
                        "request_times": valid_request_times,
                        "token_usage": valid_token_usage,
                        "daily_requests": daily_requests,
                        "last_reset_date": last_reset_date,
                    }

                logger.debug("Loaded model-specific rate limit state for %d models", len(self.model_states))

            else:
                # Old format: single model state - migrate to new format
                model_name = data.get("model", self.current_model)
                daily_requests = data.get("daily_requests", 0)

                # Load dates
                saved_date = data.get("last_reset_date")
                if saved_date:
                    saved_date = date.fromisoformat(saved_date)
                    if saved_date != tz_now().date():
                        daily_requests = 0
                        last_reset_date = tz_now().date()
                    else:
                        last_reset_date = saved_date
                else:
                    last_reset_date = tz_now().date()

                # Load request times (only keep recent ones)
                request_times = data.get("request_times", [])
                valid_request_times = deque(
                    t for t in request_times if now - t <= REQUEST_WINDOW_SECONDS
                )

                # Load token usage (only keep recent ones)
                token_usage = data.get("token_usage", [])
                valid_token_usage = deque(
                    (t, tokens) for t, tokens in token_usage if now - t <= REQUEST_WINDOW_SECONDS
                )

                self.model_states[model_name] = {
                    "request_times": valid_request_times,
                    "token_usage": valid_token_usage,
                    "daily_requests": daily_requests,
                    "last_reset_date": last_reset_date,
                }

                logger.info("Migrated old format rate limit state for model %s", model_name)

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning("Failed to load rate limit state: %s", e)
            # Reset to defaults if file is corrupted
            self.model_states.clear()

    def _save_state(self):
        """Save rate limit state to file (new model-specific format)"""
        try:
            # Convert model states to serializable format
            models_data = {}
            for model_name, state in self.model_states.items():
                models_data[model_name] = {
                    "request_times": list(state["request_times"]),
                    "token_usage": list(state["token_usage"]),
                    "daily_requests": state["daily_requests"],
                    "last_reset_date": state["last_reset_date"].isoformat(),
                }

            data = {
                "tier": self.tier,
                "current_model": self.current_model,
                "models": models_data,
                "saved_at": tz_now().isoformat(),
            }

            with open(self.state_file, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.warning("Failed to save rate limit state: %s", e)

    def _cleanup_old_records(self):
        """Remove old request records outside the time windows for current model"""
        state = self._get_model_state(self.current_model)
        now = time.time()

        # Clean up request times older than 1 minute
        while state["request_times"] and now - state["request_times"][0] > REQUEST_WINDOW_SECONDS:
            state["request_times"].popleft()

        # Clean up token usage older than 1 minute
        while state["token_usage"] and now - state["token_usage"][0][0] > REQUEST_WINDOW_SECONDS:
            state["token_usage"].popleft()

        # Reset daily counter if date changed
        current_date = tz_now().date()
        if current_date != state["last_reset_date"]:
            state["daily_requests"] = 0
            state["last_reset_date"] = current_date
            self._save_state()  # Save state after daily reset
            logger.info("Daily request counter reset for %s on %s", self.current_model, current_date)

    def _calculate_wait_time(self, estimated_tokens: int = 1000) -> float:
        """Calculate how long to wait before making the next request for current model"""
        limits = self.get_current_limits()
        state = self._get_model_state(self.current_model)
        wait_times = []

        # RPM check
        if limits.get("rpm") and len(state["request_times"]) >= limits["rpm"]:
            oldest_request = state["request_times"][0]
            wait_time = REQUEST_WINDOW_SECONDS - (time.time() - oldest_request)
            if wait_time > 0:
                wait_times.append(wait_time)

        # TPM check
        if limits.get("tpm"):
            current_tokens = sum(tokens for _, tokens in state["token_usage"])
            if current_tokens + estimated_tokens > limits["tpm"]:
                # Find the oldest token usage that we need to wait for
                tokens_to_remove = (current_tokens + estimated_tokens) - limits["tpm"]
                wait_time = 0
                for timestamp, tokens in state["token_usage"]:
                    if tokens_to_remove <= 0:
                        break
                    wait_time = REQUEST_WINDOW_SECONDS - (time.time() - timestamp)
                    tokens_to_remove -= tokens

                if wait_time > 0:
                    wait_times.append(wait_time)

        # RPD check
        if limits.get("rpd") and state["daily_requests"] >= limits["rpd"]:
            # Wait until next day using configured timezone
            now_dt = tz_now()
            tomorrow = now_dt.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            wait_time = (tomorrow - now_dt).total_seconds()
            wait_times.append(wait_time)

        return max(wait_times) if wait_times else 0

    def wait_if_needed(self, estimated_tokens: int = 1000) -> bool:
        """
        Wait if necessary to respect rate limits for current model

        Args:
            estimated_tokens: Estimated number of tokens for the request

        Returns:
            True if request can proceed, False if daily limit exceeded
        """
        with self.request_lock:
            self._cleanup_old_records()

            limits = self.get_current_limits()
            state = self._get_model_state(self.current_model)

            # Check daily limit
            if limits.get("rpd") and state["daily_requests"] >= limits["rpd"]:
                logger.error("Daily request limit (%s) exceeded for model %s", limits["rpd"], self.current_model)
                return False

            wait_time = self._calculate_wait_time(estimated_tokens)

            if wait_time > 0:
                logger.info("Rate limit reached for %s. Waiting %.2f seconds...", self.current_model, wait_time)
                time.sleep(wait_time)
                self._cleanup_old_records()

            # Record the request for current model
            now = time.time()
            state["request_times"].append(now)
            state["token_usage"].append((now, estimated_tokens))
            state["daily_requests"] += 1

            # Save state after recording request
            self._save_state()

            return True

    def get_status(self) -> dict:
        """Get current rate limit status for current model"""
        with self.request_lock:
            self._cleanup_old_records()
            limits = self.get_current_limits()
            state = self._get_model_state(self.current_model)

            current_rpm = len(state["request_times"])
            current_tpm = sum(tokens for _, tokens in state["token_usage"])
            current_rpd = state["daily_requests"]

            return {
                "tier": self.tier,
                "model": self.current_model,
                "limits": limits,
                "current": {"rpm": current_rpm, "tpm": current_tpm, "rpd": current_rpd},
                "utilization": {
                    "rpm_percent": (current_rpm / limits["rpm"] * 100) if limits.get("rpm") else 0,
                    "tpm_percent": (current_tpm / limits["tpm"] * 100) if limits.get("tpm") else 0,
                    "rpd_percent": (current_rpd / limits["rpd"] * 100) if limits.get("rpd") else 0,
                },
                "all_models": {
                    model_name: {
                        "rpm": len(model_state["request_times"]),
                        "tpm": sum(tokens for _, tokens in model_state["token_usage"]),
                        "rpd": model_state["daily_requests"],
                    }
                    for model_name, model_state in self.model_states.items()
                },
            }


# Global instance
rate_limiter = RateLimitManager()
