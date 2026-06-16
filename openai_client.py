"""openai_client.py

OpenAI 키 failover 헬퍼.

여러 API 키를 순서대로 사용 — 앞 키가 **할당량 소진(insufficient_quota)** 되면
다음 키로 자동 전환한다. (일시적 rate-limit은 호출부의 기존 재시도 로직에 맡김)

환경변수:
  OPENAI_API_KEY            기본(1순위) 키 — 예: 팀 키
  OPENAI_API_KEY_FALLBACK   2순위 키 — 예: 개인 키 (쉼표로 여러 개 가능)

사용:
  from openai_client import make_openai_client
  client = make_openai_client()
  client.chat.completions.create(...)   # OpenAI()와 동일 인터페이스
"""
from __future__ import annotations
import os

import concurrent.futures
import time

import httpx
import openai
from openai import OpenAI


def _quota_exhausted(err: Exception) -> bool:
    """insufficient_quota(할당량 소진)인지 판별. 일시적 429와 구분."""
    s = str(err).lower()
    if "insufficient_quota" in s or "exceeded your current quota" in s:
        return True
    body = getattr(err, "body", None)
    if isinstance(body, dict):
        code = (body.get("error") or {}).get("code") or body.get("code")
        if code == "insufficient_quota":
            return True
    return False


class _Completions:
    def __init__(self, parent: "FailoverOpenAI"):
        self._parent = parent

    def create(self, **kwargs):
        return self._parent._create(**kwargs)


class _Chat:
    def __init__(self, parent: "FailoverOpenAI"):
        self.completions = _Completions(parent)


# 호출이 멈추는 것 방지: 타임아웃 + 자동 재시도 (env로 조정 가능)
_TIMEOUT = float(os.environ.get("OPENAI_TIMEOUT", "40"))
_MAX_RETRIES = int(os.environ.get("OPENAI_MAX_RETRIES", "4"))
# 하드 타임아웃: httpx/SDK 타임아웃이 안 먹는 간헐적 행에 대비한 스레드 기반 강제 중단
_HARD_TIMEOUT = float(os.environ.get("OPENAI_HARD_TIMEOUT", "55"))
_EXEC = concurrent.futures.ThreadPoolExecutor(max_workers=16)


def _make_client(api_key: str | None = None) -> OpenAI:
    """keep-alive를 끈 httpx 클라이언트 + SDK 타임아웃 명시로 OpenAI 생성."""
    http = httpx.Client(
        timeout=httpx.Timeout(_TIMEOUT, connect=15.0),
        limits=httpx.Limits(max_keepalive_connections=0, max_connections=64),
    )
    # timeout을 OpenAI()에도 명시 (SDK 기본값이 httpx 설정을 덮어쓰는 것 방지)
    return OpenAI(api_key=api_key, timeout=_TIMEOUT,
                  max_retries=_MAX_RETRIES, http_client=http)


class FailoverOpenAI:
    """chat.completions.create 만 failover 지원하는 경량 래퍼."""

    def __init__(self, keys: list[str]):
        self._clients = [_make_client(api_key=k) for k in keys]
        self._idx = 0
        self.chat = _Chat(self)

    def _create(self, **kwargs):
        """하드 타임아웃 + 키 failover로 chat.completions.create 호출.

        각 시도를 별도 스레드에서 실행하고 _HARD_TIMEOUT 초 안에 안 끝나면
        버리고 재시도(새 연결). httpx/SDK 타임아웃이 안 먹는 행도 강제 탈출.
        """
        last_err = None
        for attempt in range(_MAX_RETRIES + 1):
            client = self._clients[self._idx]
            fut = _EXEC.submit(client.chat.completions.create, **kwargs)
            try:
                return fut.result(timeout=_HARD_TIMEOUT)
            except concurrent.futures.TimeoutError:
                last_err = TimeoutError(f"hard timeout {_HARD_TIMEOUT}s")
                print(f"[openai-hardtimeout] {_HARD_TIMEOUT}s 초과 → 재시도 "
                      f"{attempt + 1}/{_MAX_RETRIES + 1} (행 걸린 호출 버림)")
                continue  # fut는 백그라운드에 남지만(누수 허용) 새로 시도
            except (openai.RateLimitError, openai.AuthenticationError) as e:
                last_err = e
                if _quota_exhausted(e) or isinstance(e, openai.AuthenticationError):
                    if self._idx + 1 < len(self._clients):
                        print(f"[openai-failover] 키 #{self._idx} 소진/실패 "
                              f"({type(e).__name__}) → 키 #{self._idx + 1}로 전환")
                        self._idx += 1
                        continue
                    raise
                time.sleep(min(2 ** attempt, 10))  # 일시 rate-limit → 대기 후 재시도
                continue
            except Exception as e:
                last_err = e
                time.sleep(min(2 ** attempt, 8))
                continue
        raise last_err if last_err else RuntimeError("openai create failed")


def make_openai_client():
    """OPENAI_API_KEY (+ OPENAI_API_KEY_FALLBACK) 로 클라이언트 생성.

    fallback 키가 없으면 일반 OpenAI() 와 동일.
    """
    primary = os.environ.get("OPENAI_API_KEY", "").strip()
    fb_raw = os.environ.get("OPENAI_API_KEY_FALLBACK", "").strip()
    fallbacks = [k.strip() for k in fb_raw.split(",") if k.strip()]
    keys = ([primary] if primary else []) + fallbacks
    if not keys:
        keys = [None]  # 명시 키 없음 → env OPENAI_API_KEY 사용
    # 단일 키여도 FailoverOpenAI로 감싸서 하드 타임아웃을 항상 적용
    return FailoverOpenAI(keys)
