# ROCm long-context benchmark

Benchmark harness for **TabbyAPI + exllamav3** on Radeon (RDNA3) GPUs, used to
compare inference configs end-to-end (cold prefill, TTFT, decode throughput,
speculative-draft acceptance) at long context. Built and run on an RX 7900 XTX
(gfx1100) serving `Qwen3.8-27B-EXL3-3.5bpw`.

Results are checked in under `results/`:
- `results/prefill-192k-vs-256k.md` — `config.yml` (DFlash2/192K) vs
  `config.mtp-256k.yml` (MTP/256K) cold-prefill comparison.

## How it measures

The engine's own per-request log line is the source of truth — it reports, per
request, prompt tokens, cache-hit %, cold-prefill tokens/s, time-to-first-token,
decode tokens/s, and draft accepted/proposed. That is more accurate than timing
from the client (it excludes HTTP/queue overhead) and exposes draft behavior the
API response does not. So the harness:

1. Runs the engine with **stdout redirected to a log file**, and
2. Parses that log by **request number** (`#NN`) after each response completes.

Two things about TabbyAPI's rich logger that the parsers already handle:
- It **hard-wraps long records across physical lines even in a file** — so the
  `draft …` tail can spill to a continuation line. `report.py`/`report_decode.py`
  unwrap records (a new record starts at a timestamp) before matching.
- A cold prompt logs `none cached`, not `0% cached`; the regexes accept both.

The engine runs inside a **GNU screen session** on the host. `restart_engine.sh`
switches configs by sending the running process `Ctrl-C` through `screen -X stuff`
(python is a child of the interactive shell in that window, so the session and
shell survive) and then typing a fresh launch command with the new config and a
new log file — never attaching. It relies on `python main.py --config` being the
only such process, and on the venv being active in that shell.

## Files

| File | Purpose |
|---|---|
| `restart_engine.sh <abs_config> <abs_log>` | Stop the engine in screen (Ctrl-C) and relaunch `run_tabbyapi.sh <config>` with stdout→`<log>`. |
| `bench_longctx.py` | **Cold-prefill** client: unique random prompts, `cached=0%`, measures prefill/TTFT/decode/draft across a size ladder. |
| `bench_decode.py` | **Decode/acceptance** client: fixed natural-text prompt per size, ascending (prefix-cached), repeated reps, 512-token generations for stable decode + representative acceptance. |
| `report.py` | Renders the A-vs-B prefill table from two `bench_longctx` JSONL files (+ engine logs for draft recovery). |
| `report_decode.py` | Renders the A-vs-B decode table: median decode with min-max spread, pooled draft acceptance. |
| `results/` | Checked-in results (markdown writeups + raw JSONL). |

Both clients import shared parsing from `bench_longctx.py`; run them from
`rocm/benchmark/` (they add their own dir to `sys.path`). stdlib only.

## Configuration (env, all optional)

| Var | Default |
|---|---|
| `BENCH_MODEL` | `Qwen3.8-27B-EXL3-3.5bpw` (served model id) |
| `BENCH_URL` | `http://127.0.0.1:8096/v1/chat/completions` |
| `BENCH_KEYFILE` | `~/tabbyAPI/api_tokens.yml` (reads `api_key:`) |

Run the clients **on the host serving the engine** (localhost) so measurements
are not polluted by network round-trips.

## End-to-end: compare two configs

```sh
cd ~/exllamav3-rocm            # engine runs in the active `screen` session here

# 1) config A, with logging
rocm/benchmark/restart_engine.sh ~/tabbyAPI/config.yml /tmp/engine_A.log
# (wait for the model to load: curl -H "Authorization: Bearer $(awk -F': ' '/^api_key/{print $2}' ~/tabbyAPI/api_tokens.yml) localhost:8096/v1/models)

# 2) cold-prefill ladder A
python3 rocm/benchmark/bench_longctx.py --tag A --log /tmp/engine_A.log \
  --ladder "4096 16384 49152 98304 147456 163840 180000 190000 205000" \
  --max-tokens 128 --out /tmp/bench_A.jsonl --done /tmp/bench_A.done

# 3) switch to config B (256K) and repeat
rocm/benchmark/restart_engine.sh \
  ~/exllamav3-rocm/rocm/tabbyapi/config.mtp-256k.yml /tmp/engine_B.log
python3 rocm/benchmark/bench_longctx.py --tag B --log /tmp/engine_B.log \
  --ladder "4096 16384 49152 98304 147456 163840 180000 190000 196608 230000 256000 270000" \
  --max-tokens 128 --out /tmp/bench_B.jsonl --done /tmp/bench_B.done

# 4) compare, then restore config A
python3 rocm/benchmark/report.py /tmp/bench_A.jsonl /tmp/bench_B.jsonl \
  /tmp/engine_A.log /tmp/engine_B.log
```

Decode / draft-acceptance (the tighter, noise-resistant decode measurement):

```sh
rocm/benchmark/restart_engine.sh ~/tabbyAPI/config.yml /tmp/engine_A.log
python3 rocm/benchmark/bench_decode.py --tag A --log /tmp/engine_A.log \
  --sizes "4096 16384 49152 98304 163840" --max-tokens 512 --reps 3 \
  --out /tmp/decode_A.jsonl --done /tmp/decode_A.done
# ... same for config B -> /tmp/decode_B.* ...
python3 rocm/benchmark/report_decode.py /tmp/decode_A.jsonl /tmp/decode_B.jsonl \
  /tmp/engine_A.log /tmp/engine_B.log
```

Each client writes one JSON object per line to `--out` as it goes and touches
`--done` when finished — so you can run one in the background and watch `--out`
(e.g. `tail -f`), or drive it detached with
`setsid nohup python3 ... &` to survive ssh drops.

## Notes and caveats

- **Single stream.** `max_batch_size: 1` — only one request in flight. No other
  client should hit the endpoint during a run.
- **Cold vs warm.** `bench_longctx.py` forces cold prefill (unique prompts);
  `bench_decode.py` intentionally warms the cache (fixed prompts, ascending)
  because it measures decode, not prefill.
- **Over-limit probes** (e.g. 205K on the 192K config, 270K on the 256K config)
  are expected to return HTTP 400 `context_length_exceeded`; the clients record
  them as `error` rows and continue.
- **Decode needs long generations.** 128-token samples give noisy decode/acceptance
  (visible in the prefill table); use `bench_decode.py` (512 tokens, pooled over
  reps) for decode numbers.
- **Draft acceptance on natural text only.** `bench_longctx.py` uses random digits
  (ideal for prefill cost, meaningless for acceptance); only `bench_decode.py`
  measures acceptance representative of real workloads.
- **Restore when done**: relaunch the default config. To match the original
  (no log redirect), send the run script command directly in the screen window.
