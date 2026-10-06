'''The language-model layer: which models LedgerLens may call, in what order, and what went wrong.

The chain is Gemini > Groq > Groq (second model) > Gemini (second model): two different companies
first (the best spare during an outage), then each company's second model (same key, separate
quota bucket). A company without a real key is simply left out.
'''
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel

logger = logging.getLogger('ledgerlens')

# Single source of truth for model names. Providers retire models without warning and a retired name
# answers "model not found", so a stale value in .env is silently replaced by the current default.
GEMINI_MAIN = 'gemini-3.6-flash'
GEMINI_SPARE = 'gemini-3.5-flash-lite'   # same key, separate quota
GROQ_MAIN = 'qwen/qwen3.8-27b'
GROQ_SPARE = 'openai/gpt-oss-20b'        # same key, separate rate limits
RETIRED_GEMINI = {
    'gemini-2.5-flash', 'gemini-2.5-pro',
    'gemini-1.5-flash', 'gemini-1.5-flash-001', 'gemini-1.5-pro', 'gemini-pro',
}
RETIRED_GROQ = {   # these names now answer "model not found" on Groq
    'llama-3.3-70b-versatile', 'llama-3.1-70b-versatile', 'llama-3.1-8b-instant',
    'llama3-70b-8192', 'llama3-8b-8192', 'gemma2-9b-it', 'gemma-7b-it',
    'mixtral-8x7b-32768', 'deepseek-r1-distill-llama-70b',
    # Exists, but fails the agent: it emits a "commentary" tool call that Groq itself rejects.
    'openai/gpt-oss-120b',
}
_PLACEHOLDER_WORDS = ('your_', 'paste_', 'placeholder', 'changeme', 'xxxx')


# ----------------------------------------------------------------------------- keys and model names
def looks_like_real_key(key: str | None) -> bool:
    '''True for a non-empty value that is not one of the template placeholders.'''
    value = (key or '').strip().lower()
    return bool(value) and not any(word in value for word in _PLACEHOLDER_WORDS)


def _pick(candidate: str | None, env_name: str, default: str, retired: set[str]) -> str:
    value = (candidate or os.getenv(env_name) or '').strip()
    return default if not value or value in retired else value


def gemini_model(model_name: str | None = None) -> str:
    return _pick(model_name, 'GEMINI_MODEL', GEMINI_MAIN, RETIRED_GEMINI)


def gemini_spare_model() -> str:
    return _pick(None, 'GEMINI_BACKUP_MODEL', GEMINI_SPARE, RETIRED_GEMINI)


def groq_model(model_name: str | None = None) -> str:
    return _pick(model_name, 'GROQ_MODEL', GROQ_MAIN, RETIRED_GROQ)


def groq_spare_model() -> str:
    return _pick(None, 'GROQ_BACKUP_MODEL', GROQ_SPARE, RETIRED_GROQ)


# ----------------------------------------------------------------------------- the chain
@dataclass(frozen=True)
class Provider:
    '''One model the app may call. The key never shows up in repr() or in logs.'''
    name: str    # 'gemini', 'groq', 'groq-spare', 'gemini-spare'
    label: str   # friendly name shown in the UI
    model: str
    api_key: str = field(repr=False)


def _usable_key(passed: str | None, env_name: str) -> str | None:
    '''A key typed in the sidebar wins over the one in .env, but only if it is not a placeholder.'''
    if looks_like_real_key(passed):
        return passed.strip()
    from_env = os.getenv(env_name)
    return from_env.strip() if looks_like_real_key(from_env) else None


def active_providers(gemini_key: str | None = None, groq_key: str | None = None) -> list[Provider]:
    '''The models to try, in order. Empty when no usable key exists.'''
    google = _usable_key(gemini_key, 'GOOGLE_API_KEY')
    groq = _usable_key(groq_key, 'GROQ_API_KEY')

    gemini = Provider('gemini', 'Gemini', gemini_model(), google) if google else None
    groq_main = Provider('groq', 'Groq', groq_model(), groq) if groq else None
    gemini_second = None
    if google and gemini_spare_model() != gemini_model():
        gemini_second = Provider('gemini-spare', 'Gemini (second model)', gemini_spare_model(), google)
    groq_second = None
    if groq and groq_spare_model() != groq_model():
        groq_second = Provider('groq-spare', 'Groq (second model)', groq_spare_model(), groq)

    groq_first = os.getenv('PRIMARY_LLM', '').strip().lower() == 'groq'
    leaders = [groq_main, gemini] if groq_first else [gemini, groq_main]
    return [p for p in (*leaders, groq_second, gemini_second) if p]


def build_chat_model(provider: Provider, timeout: int = 45, max_retries: int = 1) -> BaseChatModel:
    '''Create the LangChain chat model for one provider (SDKs are imported lazily).'''
    if provider.name.startswith('gemini'):
        for noisy in ('google_genai', 'google_genai.models', 'google.genai'):
            logging.getLogger(noisy).setLevel(logging.ERROR)  # harmless "automatic function calling" notices
        from langchain_google_genai import ChatGoogleGenerativeAI
        # No temperature on purpose: Gemini 3.x uses fixed sampling and warns when one is passed.
        return ChatGoogleGenerativeAI(model=provider.model, google_api_key=provider.api_key,
                                      max_output_tokens=4096, timeout=timeout, max_retries=max_retries)
    if provider.name.startswith('groq'):
        from langchain_groq import ChatGroq
        extra: dict = {}
        if 'gpt-oss' in provider.model:
            extra['reasoning_effort'] = 'low'
        elif 'qwen' in provider.model:
            extra['reasoning_effort'] = 'none'  # thinking tokens burn Groq's small per-minute limit
        return ChatGroq(model=provider.model, api_key=provider.api_key, temperature=0.1, max_tokens=4096,
                        timeout=timeout, max_retries=max_retries, **extra)
    raise ValueError(f'Unknown provider: {provider.name}')


# ----------------------------------------------------------------------------- errors in plain words
def _text(exc: BaseException) -> str:
    return f'{type(exc).__name__} {exc}'.lower()


def _has_code(text: str, *codes: int) -> bool:
    '''True if the text holds an HTTP status code as a whole number (not inside "4013" or "0.401").'''
    return any(re.search(rf'(?<![\d.]){code}(?![\d.])', text) for code in codes)


def is_quota_error(exc: BaseException) -> bool:
    '''True for "quota / rate limit" failures.'''
    text = _text(exc)
    return _has_code(text, 429) or any(word in text for word in (
        'resource_exhausted', 'quota', 'rate limit', 'rate_limit', 'too many requests'))


def is_malformed_tool_call(exc: BaseException) -> bool:
    '''True when the model produced an unusable tool call (a known glitch of some open models).'''
    text = _text(exc)
    return any(word in text for word in ('tool_use_failed', 'tool call validation failed',
                                         'failed to call a function', 'failed_generation'))


def is_daily_limit(exc: BaseException) -> bool:
    '''True when the limit that was hit resets per day (waiting a minute will not help).'''
    text = _text(exc)
    return any(word in text for word in ('perday', 'per day', 'requests per day', 'tokens per day',
                                         '(rpd)', '(tpd)', 'daily'))


_RETRY_HINT = re.compile(r"(?:try again in|retry in|retry after|retrydelay['\": ]+)\s*([0-9hms.]+)", re.I)
_DURATION_PART = re.compile(r'(\d+(?:\.\d+)?)(ms|h|m|s)')


def retry_delay_seconds(exc: BaseException) -> float | None:
    '''Seconds the provider asked us to wait ("try again in 1m3.5s", "retry in 51s"), if it said.'''
    match = _RETRY_HINT.search(str(exc))
    if not match:
        return None
    factor = {'ms': 0.001, 's': 1, 'm': 60, 'h': 3600}
    parts = _DURATION_PART.findall(match.group(1).lower())
    return sum(float(number) * factor[unit] for number, unit in parts) if parts else None


def describe_error(exc: BaseException) -> str:
    '''Turn an upstream exception into one short, safe sentence (never the raw message).'''
    text = _text(exc)
    if _has_code(text, 401, 403) or any(word in text for word in (
            'api key not valid', 'api_key_invalid', 'invalid api key', 'invalid_api_key',
            'unauthorized', 'permission_denied', 'authentication')):
        return 'the API key was not accepted (it may be invalid, expired or not allowed)'
    if is_quota_error(exc):
        return 'the free usage limit has been reached'
    if _has_code(text, 404) or any(word in text for word in (
            'not_found', 'not found', 'model_not_found', 'decommissioned', 'does not exist')):
        return 'the model name was not recognised for this key'
    if is_malformed_tool_call(exc):
        return 'the model replied in an unusable format (a temporary glitch)'
    if _has_code(text, 502, 503, 504) or any(word in text for word in (
            'timeout', 'timed out', 'deadline', 'connection', 'connecterror', 'network',
            'name resolution', 'unreachable', 'unavailable', 'high demand', 'overloaded')):
        return 'the service is busy or cannot be reached right now (outage, timeout or network problem)'
    return 'an unexpected error happened (details are in the terminal window)'


def redact(text: str, secrets: list[str]) -> str:
    '''Replace every secret longer than 6 characters with *** (upstream errors sometimes echo keys).'''
    for secret in secrets:
        if secret and len(secret) > 6:
            text = text.replace(secret, '***')
    return text


# ----------------------------------------------------------------------------- quota-free status check
def check_provider(provider: Provider, timeout: int = 15) -> tuple[bool, str]:
    '''Ask only for account metadata (never a generation), so the status light costs no free quota.'''
    try:
        if provider.name.startswith('gemini'):
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=provider.api_key,
                                  http_options=types.HttpOptions(timeout=timeout * 1000))
            client.models.get(model=provider.model)
        else:
            from groq import Groq
            listing = Groq(api_key=provider.api_key, timeout=timeout, max_retries=0).models.list()
            if provider.model not in {item.id for item in listing.data}:
                return False, 'the model name was not recognised for this key'
        return True, 'key works'
    except Exception as exc:  # noqa: BLE001 - any failure just means "not usable right now"
        logger.warning('Status check for %s failed: %s', provider.name,
                       redact(str(exc), [provider.api_key])[:300])
        return False, describe_error(exc)
