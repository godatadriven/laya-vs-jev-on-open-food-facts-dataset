"""Send the sample to one system and keep every raw response, then build one results table from them.

A run is a folder `data/runs/<run>/` holding, for each system:

- `<system>.jsonl`, the raw log: one line per product and attempt, with the raw response or the error;
- `<system>.json`, the manifest: what the run is pinned to, and every invocation with its timings.

Rerunning a run sends only the products without a response yet, so a stopped run continues where it
stopped. A failed request is retried by the next invocation, and its failure stays in the log.
"""

import asyncio
import datetime as dt
import json
import os
import time
from dataclasses import dataclass, field
from itertools import batched
from pathlib import Path
from typing import TextIO

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv

from harness.client import Reply, RequestFailed, SystemOne
from harness.questions import QUESTIONS, probabilities, questions_sha256, read_answer
from harness.training import CHECKPOINT, WEIGHTS, sha256

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "data" / "sample.parquet"
RUNS = ROOT / "data" / "runs"
LAYA_VERSION = "0.4.18"  # the release `just laya` serves; /health does not report it
ABORT_AFTER = 5  # requests failing in a row: the system is down, so stop instead of logging failures
PROGRESS_EVERY_S = 30


@dataclass(frozen=True)
class System:
    name: str
    base_url: str
    model: str
    api_key_env: str | None = None
    timeout_s: float = 120
    chunk: int = 1  # states per request; more than one goes through the batch route
    batch_options: dict = field(default_factory=dict)
    workers: int = 1  # requests in flight
    per_minute: float | None = None  # cap on request starts, retries included
    max_attempts: int = 6
    backoff_s: float = 1.0
    laya: bool = False  # served by Laya, which reports its device and checkpoint on /health
    checkpoint: Path | None = None  # a local checkpoint that Laya serves under `model`, instead of a published one

    @property
    def route(self) -> str:
        return "/v1/systemone/batch" if self.chunk > 1 else "/v1/systemone"


SYSTEMS = {
    # Compute-bound on MPS: 64 states per call, sorted by length into passes of 16, measured fastest
    "laya": System(
        "laya",
        "http://127.0.0.1:8765",
        "english",
        timeout_s=600,
        chunk=64,
        batch_options={"sort_by_length": True, "batch_size": 16},
        laya=True,
    ),
    # The same, fine-tuned on the calibration split; `just laya-ft` serves it beside the published one
    "laya-ft": System(
        "laya-ft",
        "http://127.0.0.1:8767",
        "laya-ft",
        timeout_s=600,
        chunk=64,
        batch_options={"sort_by_length": True, "batch_size": 16},
        laya=True,
        checkpoint=CHECKPOINT,
    ),
    # One state per request: a few in flight, at half the published limit of 1200 requests per minute
    "jev": System(
        "jev",
        "https://api.typesafe.ai",
        "jev-1.14.0",
        api_key_env="JEV_API_KEY",
        workers=4,
        per_minute=600,
    ),
}


@dataclass(frozen=True)
class Product:
    id: int
    state: str


def load_products(split: str | None = None, limit: int | None = None) -> list[Product]:
    """The sample rows of `split`, or of both splits, in id order; the first `limit` of them."""
    query = "SELECT id, state FROM read_parquet(?)"
    params: list = [str(SAMPLE)]
    if split:
        query += " WHERE split = ?"
        params.append(split)
    query += " ORDER BY id"
    if limit:
        query += f" LIMIT {int(limit)}"
    rows = duckdb.connect().execute(query, params).fetchall()
    return [Product(id, state) for id, state in rows]


def read_log(path: Path) -> list[dict]:
    """Every line of a raw log. A last line cut off by a hard stop is removed from the file."""
    if not path.exists():
        return []
    head, _, tail = path.read_text().rpartition("\n")
    if tail:
        path.write_text(head + "\n" if head else "")
    return [json.loads(line) for line in head.splitlines() if line]


def now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


async def resolve_version(system: System, client: SystemOne) -> tuple[str, str | None, dict | None]:
    """The model version a system answers with, Laya's checkpoint revision, and Laya's `/health`.

    A local checkpoint has no commit, so the digest of its weights stands in for one.
    """
    if not system.laya:
        return system.model, None, None  # pinned by name; every response is checked against it
    health = await client.get("/health")
    if health["device"] == "cpu":  # every invocation keeps /health, so the device of each answer is on record
        raise RuntimeError(f"the Laya on {system.base_url} fell back to the CPU")
    if system.model not in health["loaded"]:
        raise RuntimeError(f"the Laya on {system.base_url} does not serve {system.model}")
    if system.checkpoint and not (system.checkpoint / WEIGHTS).exists():
        raise RuntimeError(f"no checkpoint in {system.checkpoint.relative_to(ROOT)}: install it with `just ft-install`")
    revision = sha256(system.checkpoint / WEIGHTS) if system.checkpoint else health["revisions"][system.model]
    return f"laya {LAYA_VERSION}, {system.model}@{revision[:7]}", revision, health


def open_manifest(path: Path, pinned: dict, questions: dict) -> dict:
    if not path.exists():
        return {"pinned": pinned, "questions": questions, "invocations": []}
    manifest = json.loads(path.read_text())
    changed = sorted(key for key in pinned if manifest["pinned"].get(key) != pinned[key])
    if changed:
        raise RuntimeError(f"{path.relative_to(ROOT)} is pinned to another {', '.join(changed)}; start a new run name")
    return manifest


class Writer:
    """Appends one log line per product, keeps count, prints progress, and stops when a system is down."""

    def __init__(self, file: TextIO, system: str, model_version: str, total: int):
        self.file = file
        self.system = system
        self.model_version = model_version
        self.total = total
        self.done = self.failed = self.failed_in_a_row = 0
        self.start = self.printed = time.monotonic()

    def line(self, product: Product, *, response, error, latency_ms, attempts, call) -> None:
        self.file.write(
            json.dumps(
                {
                    "id": product.id,
                    "system": self.system,
                    "model_version": self.model_version,
                    "at": now(),
                    "attempts": attempts,
                    "latency_ms": latency_ms,
                    "call": call,
                    "error": error,
                    "response": response,
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    def answered(self, products: list[Product], responses: list[dict], reply: Reply, call: dict) -> None:
        for product, response in zip(products, responses, strict=True):
            self.line(
                product,
                response=response,
                error=None,
                latency_ms=round(reply.ms / len(products), 1),
                attempts=reply.attempts,
                call=call,
            )
        self.failed_in_a_row = 0
        self.tick(len(products))

    def failed_request(self, products: list[Product], failure: RequestFailed, call: dict) -> None:
        for product in products:
            self.line(product, response=None, error=str(failure), latency_ms=None, attempts=failure.attempts, call=call)
        self.failed += len(products)
        self.failed_in_a_row += 1
        self.tick(len(products))
        if self.failed_in_a_row >= ABORT_AFTER:
            raise RuntimeError(f"{self.system}: {ABORT_AFTER} requests in a row failed, the last with {failure}")

    def tick(self, products: int) -> None:
        self.file.flush()
        self.done += products
        if time.monotonic() - self.printed >= PROGRESS_EVERY_S:
            self.printed = time.monotonic()
            print(self.progress(), flush=True)

    def progress(self) -> str:
        elapsed = time.monotonic() - self.start
        rate = self.done / elapsed if elapsed else 0.0
        eta = (self.total - self.done) / rate if rate else 0.0
        return (
            f"{self.system}: {self.done:,} of {self.total:,} products, {self.failed:,} failed, "
            f"{rate:.1f} products/s, {duration(elapsed)} so far, {duration(eta)} to go"
        )


def duration(seconds: float) -> str:
    if seconds < 120:
        return f"{seconds:.0f} s"
    minutes = round(seconds / 60)
    return f"{minutes} min" if minutes < 60 else f"{minutes // 60} h {minutes % 60:02d} min"


async def send_batches(client: SystemOne, system: System, questions: dict, todo: list[Product], writer: Writer) -> None:
    # Warm-up, not logged: the first call after a start spends about 13 s compiling MPS kernels
    await client.decide_batch([todo[0].state], questions, **system.batch_options)
    for chunk in map(list, batched(todo, system.chunk)):
        call = {"route": system.route, "first_id": chunk[0].id, "states": len(chunk)}
        try:
            reply = await client.decide_batch([product.state for product in chunk], questions, **system.batch_options)
        except RequestFailed as failure:
            writer.failed_request(chunk, failure, call)
            continue
        # The response is split into one line per product; nothing else in it may be lost on the way
        if set(reply.json) != {"results", "total_usage"} or len(reply.json["results"]) != len(chunk):
            raise RuntimeError(f"{system.name}: unexpected batch response with keys {sorted(reply.json)}")
        call |= {"ms": round(reply.ms, 1), "total_usage": reply.json["total_usage"]}
        writer.answered(chunk, reply.json["results"], reply, call)


async def send_singles(
    client: SystemOne,
    system: System,
    questions: dict,
    todo: list[Product],
    writer: Writer,
    model_version: str,
) -> None:
    queue = iter(todo)  # shared by the workers; taking the next product never awaits

    async def worker() -> None:
        for product in queue:
            call = {"route": system.route, "states": 1}
            try:
                reply = await client.decide(product.state, questions)
            except RequestFailed as failure:
                writer.failed_request([product], failure, call)
                continue
            if reply.json.get("model") != model_version:
                raise RuntimeError(
                    f"{system.name} answered as {reply.json.get('model')}, the run is pinned to {model_version}"
                )
            writer.answered([product], [reply.json], reply, call | {"ms": round(reply.ms, 1)})

    try:
        async with asyncio.TaskGroup() as group:  # one worker failing stops the others
            for _ in range(system.workers):
                group.create_task(worker())
    except ExceptionGroup as errors:
        raise errors.exceptions[0]


async def run(
    system: str | System,
    run: str = "full",
    *,
    split: str | None = None,
    limit: int | None = None,
    questions: dict = QUESTIONS,
) -> dict:
    """Send the products of `split` (both splits without one), the first `limit` of them, to `system`.

    `questions` defaults to request.json; another wording or option order goes in a run of its own.
    Returns this invocation's summary, which is also appended to the run's manifest.
    """
    system = SYSTEMS[system] if isinstance(system, str) else system
    load_dotenv(ROOT / ".env")
    api_key = os.environ[system.api_key_env] if system.api_key_env else None
    folder = RUNS / run
    folder.mkdir(parents=True, exist_ok=True)
    products = load_products(split, limit)

    async with SystemOne(
        system.base_url,
        system.model,
        api_key,
        timeout=system.timeout_s,
        per_minute=system.per_minute,
        max_attempts=system.max_attempts,
        backoff_s=system.backoff_s,
    ) as client:
        model_version, revision, health = await resolve_version(system, client)
        pinned = {
            "system": system.name,
            "base_url": system.base_url,
            "model": system.model,
            "model_version": model_version,
            "revision": revision,
            "checkpoint": str(system.checkpoint.relative_to(ROOT)) if system.checkpoint else None,
            "route": system.route,
            "chunk": system.chunk,
            "batch_options": system.batch_options,
            "questions_sha256": questions_sha256(questions),
            "split": split,
            "limit": limit,
        }
        manifest_path = folder / f"{system.name}.json"
        manifest = open_manifest(manifest_path, pinned, questions)
        log_path = folder / f"{system.name}.jsonl"
        answered = {line["id"] for line in read_log(log_path) if line["response"] is not None}
        todo = [product for product in products if product.id not in answered]
        invocation = {
            "started": now(),
            "products": len(products),
            "to_send": len(todo),
            "workers": system.workers,
            "per_minute": system.per_minute,
            "health": health,
        }
        print(f"{system.name}: {len(todo):,} of {len(products):,} products to send to {model_version}", flush=True)

        start = time.perf_counter()
        with log_path.open("a") as file:
            writer = Writer(file, system.name, model_version, len(todo))
            try:
                if todo and system.chunk > 1:
                    await send_batches(client, system, questions, todo, writer)
                elif todo:
                    await send_singles(client, system, questions, todo, writer, model_version)
                if revision and (await resolve_version(system, client))[1] != revision:
                    raise RuntimeError(f"{system.name} changed its checkpoint during the run")
            finally:
                invocation |= {
                    "finished": now(),
                    "seconds": round(time.perf_counter() - start, 1),
                    "sent": writer.done,
                    "failed": writer.failed,
                }
                manifest["invocations"].append(invocation)
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
                if todo:
                    print(writer.progress(), flush=True)
    return invocation


RESULTS_SCHEMA = pa.schema(
    [
        ("id", pa.int64()),
        ("split", pa.string()),
        ("system", pa.string()),
        ("model_version", pa.string()),
        ("question", pa.string()),
        ("answer", pa.string()),
        ("probs", pa.map_(pa.string(), pa.float64())),
        ("latency_ms", pa.float64()),  # per product; a batch call's time is shared by its products
        ("input_tokens", pa.int64()),  # per product, counted by each system's own tokenizer
        ("truncated", pa.bool_()),  # Laya only: the state did not fit next to this question
        ("state_tokens_dropped", pa.int64()),  # Laya only
        ("attempts", pa.int64()),
        ("error", pa.string()),
    ]
)


def final_lines(path: Path) -> dict[int, dict]:
    """The line that counts for each product: its response, or its last failure if it never got one."""
    final: dict[int, dict] = {}
    for line in read_log(path):
        if line["id"] not in final or final[line["id"]]["response"] is None:
            final[line["id"]] = line
    return final


def build(run: str = "full") -> Path:
    """Turn the raw logs of `run` into `results.parquet`: one row per product, system and question."""
    folder = RUNS / run
    split_of, state_of = {}, {}
    for id, split, state in (
        duckdb.connect().execute("SELECT id, split, state FROM read_parquet(?)", [str(SAMPLE)]).fetchall()
    ):
        split_of[id], state_of[id] = split, state

    rows = []
    for manifest_path in sorted(folder.glob("*.json")):
        manifest = json.loads(manifest_path.read_text())
        pinned, questions = manifest["pinned"], manifest["questions"]
        final = final_lines(folder / f"{pinned['system']}.jsonl")
        if not final:
            continue
        versions = {line["model_version"] for line in final.values()}
        if len(versions) != 1:
            raise RuntimeError(f"{pinned['system']} answered as {sorted(versions)} in one run")
        ids = sorted(final)
        tokens, rooms = {}, {}
        if pinned["revision"]:
            from harness.truncation import Truncation  # imports torch, so only when Laya is in the run

            checkpoint = pinned.get("checkpoint") and ROOT / pinned["checkpoint"]
            if checkpoint and sha256(checkpoint / WEIGHTS) != pinned["revision"]:
                raise RuntimeError(f"{pinned['checkpoint']} holds another checkpoint than the run answered with")
            counter = Truncation(pinned["revision"], LAYA_VERSION, checkpoint)
            tokens = dict(zip(ids, counter.state_tokens([state_of[id] for id in ids]), strict=True))
            rooms = {question: counter.room(questions[question]) for question in questions}

        for id in ids:
            line = final[id]
            response = line["response"]
            for question in questions:
                probs = probabilities(response["answers"][question], question) if response else None
                dropped = max(0, tokens[id] - rooms[question]) if tokens else None
                rows.append(
                    {
                        "id": id,
                        "split": split_of[id],
                        "system": pinned["system"],
                        "model_version": line["model_version"],
                        "question": question,
                        "answer": read_answer(probs) if probs else None,
                        "probs": list(probs.items()) if probs else None,
                        "latency_ms": line["latency_ms"],
                        "input_tokens": response["usage"]["input_tokens"] if response else None,
                        "truncated": dropped > 0 if tokens else None,
                        "state_tokens_dropped": dropped,
                        "attempts": line["attempts"],
                        "error": line["error"],
                    }
                )

    path = folder / "results.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=RESULTS_SCHEMA), path)
    return path
