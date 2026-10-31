"""One client for the `/v1/systemone` protocol: `base_url`, `model` and the API key are all that differ."""

import asyncio
import email.utils
import time
from dataclasses import dataclass
from typing import Self

import httpx

# Worth another attempt: the server was busy, overloaded or briefly away. Anything else is final
RETRY_STATUS = {408, 429, 500, 502, 503, 504}
MAX_RETRY_AFTER_S = 120


@dataclass
class Reply:
    """A successful response, how long its attempt took, and how many attempts it needed."""

    json: dict
    ms: float
    attempts: int


class RequestFailed(Exception):
    """A request that failed on every attempt, or in a way another attempt cannot fix."""

    def __init__(self, problem: str, attempts: int):
        super().__init__(problem)
        self.attempts = attempts


class Pace:
    """Spaces request starts at least `60 / per_minute` seconds apart, across every worker."""

    def __init__(self, per_minute: float):
        self.interval = 60 / per_minute
        self.next_start = 0.0
        self.lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self.lock:
            now = time.monotonic()
            delay = self.next_start - now
            self.next_start = max(now, self.next_start) + self.interval
        if delay > 0:
            await asyncio.sleep(delay)


class SystemOne:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        *,
        timeout: float = 120,
        per_minute: float | None = None,
        max_attempts: int = 6,
        backoff_s: float = 1.0,
    ):
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.http = httpx.AsyncClient(base_url=base_url, headers=headers, timeout=timeout)
        self.model = model
        self.pace = Pace(per_minute) if per_minute else None
        self.max_attempts = max_attempts
        self.backoff_s = backoff_s

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc) -> None:
        await self.http.aclose()

    async def get(self, path: str) -> dict:
        response = await self.http.get(path)
        response.raise_for_status()
        return response.json()

    async def decide(self, state: str, questions: dict) -> Reply:
        return await self.post("/v1/systemone", {"state": state, "questions": questions})

    async def decide_batch(self, states: list[str], questions: dict, **options) -> Reply:
        """`options` are the batch route's own fields, such as `sort_by_length` and `batch_size`."""
        return await self.post("/v1/systemone/batch", {"states": states, "questions": questions, **options})

    async def post(self, path: str, body: dict) -> Reply:
        """POST `body` with the model added, retrying what is worth retrying with exponential backoff."""
        for attempt in range(1, self.max_attempts + 1):
            if self.pace:
                await self.pace.wait()
            start = time.perf_counter()
            try:
                response = await self.http.post(path, json={"model": self.model, **body})
            except httpx.TransportError as error:
                problem, wait_s = f"{type(error).__name__}: {error}", None
            else:
                if response.is_success:
                    return Reply(response.json(), (time.perf_counter() - start) * 1000, attempt)
                problem = f"HTTP {response.status_code}: {response.text[:500]}"
                if response.status_code not in RETRY_STATUS:
                    raise RequestFailed(problem, attempt)
                wait_s = retry_after_s(response)
            if attempt < self.max_attempts:
                await asyncio.sleep(wait_s if wait_s is not None else self.backoff_s * 2 ** (attempt - 1))
        raise RequestFailed(problem, self.max_attempts)


def retry_after_s(response: httpx.Response) -> float | None:
    """The wait a 429 or 503 asks for, in seconds or as an HTTP date, capped at MAX_RETRY_AFTER_S."""
    value = response.headers.get("retry-after")
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            seconds = email.utils.parsedate_to_datetime(value).timestamp() - time.time()
        except (TypeError, ValueError):
            return None
    return min(max(seconds, 0.0), MAX_RETRY_AFTER_S)
