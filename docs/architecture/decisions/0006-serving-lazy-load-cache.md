# ADR 0006 — Serving layer loads only the default model at startup, lazy-loads the rest

- **Status:** Accepted — 2026-04-25.
- **Deciders:** Project author; Stage 12 lead agent; `@minimalist` Phase-1 critic (plan [`12-serving.md`](../../plans/completed/12-serving.md) D7 — *single highest-leverage cut*).
- **Related:** [`layers/serving.md`](../layers/serving.md), [`layers/registry.md`](../layers/registry.md), [`src/bristol_ml/serving/app.py`](../../../src/bristol_ml/serving/app.py), [`docs/lld/research/12-serving-scope-diff.md`](../../lld/research/12-serving-scope-diff.md), Stage 12 plan §1 D7.

## Context

Stage 12 ships a single `POST /predict` endpoint backed by the Stage 9 registry. The endpoint accepts an optional `run_id` field; absent the field, the service uses a "default model" resolved at startup as the lowest-MAE registered run (D6).

The lead's pre-Phase-1 draft loaded **every registered run** into memory at startup, indexed by `run_id`, so any subsequent `run_id` value would hit the in-memory cache. This was structurally simpler in the request-handling path (no branching on cache miss) but non-trivial at startup: `N` model loads, `N` failure paths, `N` artefacts on disk that must all parse, all consume memory, and all be tested.

The `@minimalist` Phase-1 critic flagged this as `PLAN POLISH` in [`docs/lld/research/12-serving-scope-diff.md`](../../lld/research/12-serving-scope-diff.md):

> *"AC-5 / AC-3 only require one model loaded; eager loading turns one error path (default model load fails) into N error paths with N tests; startup latency scales with registry size."*

The Scope Diff named this row the *single highest-leverage cut* — the change that yielded the largest reduction in surface area for the smallest plan edit.

The decision is load-bearing because it shapes the request handler's call path (lazy-load cache check), the lifespan's shutdown semantics (cache eviction), and the failure mode the operator sees on the first request to a non-default `run_id` (lazy-load latency on the first call only).

Three project constraints narrow the choice:

1. **DESIGN §2.2.4** (complexity is earned) — eager loading is "machinery whose benefit is not concrete" when the AC set requires only one model loaded.
2. **AC-1** ("starts on a clean machine without configuration beyond pointing at a registry location") — startup must succeed deterministically against a non-trivial registry. With eager loading, a single corrupted artefact in `data/registry/` would prevent startup; with lazy loading, only a corrupted *default* artefact does.
3. **AC-3** (prediction parity) — both strategies satisfy AC-3 identically; the cache layer is invisible to the parity assertion.

## Decision

The lifespan loads only the **default model** (lowest-MAE run, D6) into `app.state.loaded: dict[str, Model]`, keyed by `run_id`. Subsequent requests with a non-default `run_id` lazy-load the model on first use and cache it in the same dict; subsequent requests with that `run_id` reuse the cached instance. Cache lifetime is the lifespan; the dict is cleared at shutdown so successive `TestClient` contexts in the test suite do not share state and so torch / large model state does not leak between successive ASGI restarts.

```python
# bristol_ml/serving/app.py — request handler core
run_id = req.run_id or app.state.default_run_id
if run_id not in app.state.loaded:
    try:
        app.state.loaded[run_id] = registry.load(
            run_id, registry_dir=app.state.registry_dir
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=...) from exc
model = app.state.loaded[run_id]
```

The cache is intentionally a flat dict — no LRU eviction, no TTL, no concurrency control beyond the GIL. The serving layer is single-process and the registry is bounded (~10s of runs in any pedagogical use); a more elaborate cache buys nothing.

## Consequences

- **One startup error path, not N.** A corrupted `data/registry/<other-run>/` does not prevent startup; only a corrupted *default* run does. The operator sees a clear `RuntimeError` naming the default run id rather than a generic "one of N runs failed to load" message.
- **Startup latency is `O(1)`** in registry size — only the default model is materialised. This matters as the registry grows: by Stage 18 a project running drift monitoring may have 50+ runs, and eager-loading would mean a 50× cold start.
- **First request to a non-default `run_id` is slower** than subsequent requests to the same `run_id`, because the first request includes the load cost. The latency window in the structured log (D11) deliberately wraps `model.predict` only — not the lazy-load — so the recorded `latency_ms` reflects steady-state cost rather than first-call cost. This is the load-bearing reason the latency window is scoped narrowly: a Stage 18 drift consumer reading `latency_ms` should see a comparable per-call number across requests, not a first-call outlier.
- **404 surface widens by one path.** Eager loading would 404 on unknown `run_id` because the dict lookup misses; lazy loading 404s on unknown `run_id` because `registry.load` raises `FileNotFoundError`. The handler converts both into the same HTTP 404 with a `detail` naming the missing run id and the registry directory; AC-2 ("clear error on invalid input") is satisfied identically.
- **Cache is process-local; no cross-worker sharing.** If the serving layer is ever scaled to multiple uvicorn workers, each worker has its own cache and each pays the lazy-load cost on its first encounter with a new `run_id`. This is acceptable for the single-process pedagogical demo; a multi-worker production deployment would warrant a shared cache (Redis, memcached) or cache warming on startup, owned by a future stage.
- **Test surface is smaller.** One eager-load happy path test, one lazy-load happy path test, one lazy-load cache-hit test (`test_lazy_load_caches_run_id_after_first_request`) — three integration tests cover the cache contract. The eager-loading variant would have required N parametrised happy-path tests plus N parametrised "one of N corrupted" failure tests.

## Alternatives considered

- **Eager-load every registered run at startup.** Rejected per the Scope Diff — `PLAN POLISH`. AC-set requires only one model loaded; eager loading turns one error path into N, scales startup latency with registry size, and inflates the test surface for no AC-grounded benefit. Revisit if Stage 18 introduces an AC like *"every registered run must be queryable within X ms of startup"*; that would make eager loading a requirement rather than polish.
- **Load-on-every-request, no cache.** Rejected. Each request would pay the full load cost; for `nn_temporal` and `sarimax` this is in the tens-to-hundreds of milliseconds (vs. sub-millisecond predict). The structured log's `latency_ms` would be dominated by load cost rather than predict cost, making it useless for drift monitoring's per-call latency signal. The cache costs ~5 lines of code; the benefit is a 100×+ improvement in steady-state latency for non-default runs.
- **LRU cache with a fixed capacity.** Rejected. The registry is bounded in pedagogical use; a flat dict with no eviction is structurally simpler and the memory cost is bounded by the registry size, which is the operator's explicit choice. If a future stage introduces unbounded run growth (e.g. an online-learning pipeline that registers a new run per training cycle), the LRU is a one-line `functools.lru_cache` retrofit.
- **TTL-based cache with periodic eviction.** Rejected. Adds a background task and a clock dependency for no AC-grounded benefit; the lifespan-scoped dict is cleared automatically at shutdown, which is the only eviction the project needs.

## Supersession

If a future stage requires multi-worker serving or a query pattern where every registered run must be queryable within a bounded time of startup, this ADR should be superseded by a new one that records the harder cache requirements (shared cache, cache warming, capacity limits). The current decision is correct for the single-process pedagogical demo and the Stage 18 drift consumer but should not be assumed to extend to multi-worker production deployments.

## References

- [`docs/plans/completed/12-serving.md`](../../plans/completed/12-serving.md) §1 D7 — the lead's plan record and the Scope Diff cross-reference.
- [`docs/lld/research/12-serving-scope-diff.md`](../../lld/research/12-serving-scope-diff.md) §5 — the `@minimalist` "single highest-leverage cut" sentence.
- [`docs/lld/research/12-serving-domain.md`](../../lld/research/12-serving-domain.md) §R7 — FastAPI lifespan pattern research.
- [`src/bristol_ml/serving/app.py`](../../../src/bristol_ml/serving/app.py) — the implementation, with inline D7 citations on the lazy-load branch and the latency-window scope.
- [`docs/architecture/layers/serving.md`](../layers/serving.md) — the layer doc that consumes this decision.
