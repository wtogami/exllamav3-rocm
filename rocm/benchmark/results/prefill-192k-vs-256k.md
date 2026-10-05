# Long-context prefill: config.yml (DFlash2 / 192K) vs config.mtp-256k.yml (MTP / 256K)

Cold-prefill comparison of the two shipped ROCm TabbyAPI configs on a single
Radeon RX 7900 XTX (gfx1100), serving `Qwen3.8-27B-EXL3-3.5bpw`.

| | A — `config.yml` | B — `config.mtp-256k.yml` |
|---|---|---|
| Context cap | 196,608 (192K) | 262,144 (256K) |
| Draft method | DFlash2 **separate draft model** | MTP head (built into the model) |
| Draft KV cache | Q4 | Q8 |
| Everything else | identical (Q8 main, hybrid GatedDeltaNet) | identical |

This is a **config-vs-config** comparison: B changes three things at once (draft
method, context size, draft-KV precision), so differences are not attributable
to any single knob — except that prefill is provably context/memory-bound and
draft-agnostic (see below).

## Method

- Client: `bench_longctx.py`. Each request is a **cold, unique** prompt (random
  9-digit numbers behind a fresh nonce), so `cached=0%` on every row — true
  worst-case long-document ingestion cost, no KV reuse.
- Metrics come from TabbyAPI's own per-request log line (redirected to a file
  via `restart_engine.sh`), parsed in `bench_longctx.py` / `report.py`. This is
  more accurate than client-side timing (excludes HTTP overhead) and gives
  cold prefill rate, TTFT, decode rate, and draft acceptance directly.
- Run from the engine host (localhost) to remove network variance.
- `max_tokens=128`. Date: 2026-10-04.

## Results

| Context | A prompt | A prefill T/s | A TTFT | A decode | A draft | B prompt | B prefill T/s | B TTFT | B decode | B draft |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4K   |  4,162 | 1169 | 3.6s  | 97  | 38% |  4,166 | 1151 | 3.6s  | 88  | 57% |
| 16K  | 16,442 | 1158 | 14.2s | 116 | 51% | 16,454 | 1143 | 14.4s | 85  | 57% |
| 49K  | 49,213 | 1030 | 47.8s | 90  | 43% | 49,213 | 1009 | 48.8s | 105 | 89% |
| 98K  | 98,346 |  873 | 112.6s| 103 | 61% | 98,366 |  851 | 115.6s| 93  | 89% |
| 147K | 147,504|  757 | 194.9s| 107 | 75% | 147,513|  735 | 200.6s| 82  | 89% |
| 164K | 163,878|  725 | 226.1s| 102 | 75% | 163,891|  705 | 232.6s| 79  | 89% |
| 180K | 180,041|  690 | 260.8s| 73  | 53% | 180,051|  675 | 266.6s| 68  | 79% |
| 190K | 190,036|  679 | 280.0s| 95  | 75% | 190,052|  659 | 288.6s| 59  | 66% |
| 196K | —      | —    | —     | —   | —   | 196,662|  648 | 303.6s| 47  | 47% |
| 230K | —      | —    | —     | —   | —   | 230,053|  597 | 385.5s| 68  | 89% |
| 256K | —      | —    | —     | —   | —   | 256,054|  563 | 454.8s| 46  | 57% |
| 205K | 400 over-context | | | | | — | | | | |
| 270K | — | | | | | 400 over-context | | | | |

`—` for A above 192K: the request is rejected with
`context_length_exceeded` (A's hard wall is 196,608; B's is 262,144).

## Findings

1. **Cold prefill and TTFT are effectively identical** — B is within 1-3% of A at
   every matched size, and both fall ~41-42% from 4K→190K (1169→679 vs
   1151→659 T/s). The draft method does not affect prefill; B's tiny consistent
   lag is its larger Q8 KV + 256K cache allocation.
2. **Context ceiling is the real difference**: A hard-stops at 196,608; B runs to
   262,144 (+33% context). If a workload needs >192K tokens, B is the only option.
3. **B uses LESS VRAM than A despite the larger cache** — ~23.9 GB vs ~25.1 GB of
   24 GB. A carries an entire second 27B draft model (DFlash2) in VRAM; B's MTP
   is a tiny built-in head. MTP pays for longer context by dropping that model.
   System RAM barely moved (KV cache is statically preallocated on-GPU).
4. **MTP draft acceptance is clearly higher** — ~57-89% (B) vs ~38-75% (A) —
   repeatably. The model's own MTP head predicts the target's output more
   agreeably than a bolt-on draft model.
5. **Decode is NOT separable in this test.** With `max_tokens=128` the decode
   deltas (−38%..+17%) are measurement noise, not signal — see the dedicated
   decode benchmark in `decode-*.md` for the proper measurement.

## Recommendation

Prefill/TTFT are a tie (draft-agnostic), so **choose by workload**. The follow-up
decode study (`decode-192k-vs-256k.md`) measured generation speed properly and
found **A (DFlash2) decodes faster than B at every context depth**, so this
superseded an earlier tentative "B is better by default" read. Net:

- **Need > 192K tokens** → **B** (`config.mtp-256k.yml`) is the only option (and
  uses slightly less VRAM).
- **Within 192K and you care about decode speed / agent responsiveness** →
  **A** (`config.yml`, DFlash2) is faster and holds up far better deep in context.
- **Prefill-dominated (RAG, long-doc ingest, few output tokens)** → equivalent;
  pick for context ceiling.

## Reproduce

```sh
# on the engine host, with the engine's stdout redirected to a log file:
rocm/benchmark/restart_engine.sh ~/tabbyAPI/config.yml /tmp/engine_A.log
python3 rocm/benchmark/bench_longctx.py --tag A --log /tmp/engine_A.log \
  --ladder "4096 16384 49152 98304 147456 163840 180000 190000 205000" \
  --max-tokens 128 --out /tmp/bench_A.jsonl --done /tmp/bench_A.done
# ... repeat for config.mtp-256k.yml -> /tmp/bench_B.* ...
python3 rocm/benchmark/report.py /tmp/bench_A.jsonl /tmp/bench_B.jsonl \
  /tmp/engine_A.log /tmp/engine_B.log
```

Raw per-request data: `prefill_A_config-dflash2-192k.jsonl`,
`prefill_B_config-mtp-256k.jsonl` (one JSON object per line; see
`bench_longctx.py` for field meanings).
