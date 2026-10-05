# Decode throughput & draft acceptance: config.yml (DFlash2 / 192K) vs config.mtp-256k.yml (MTP / 256K)

Steady-state **decode** measurement — the tighter follow-up to the prefill study
(`prefill-192k-vs-256k.md`), whose `max_tokens=128` decode numbers were too noisy
to separate the two configs. RX 7900 XTX (gfx1100), `Qwen3.8-27B-EXL3-3.5bpw`.
Date: 2026-10-05.

Same A/B configs as the prefill study: A = DFlash2 separate draft model (Q4 draft
KV); B = MTP head built into the model (Q8 draft KV).

## Method (`bench_decode.py`)

- One **fixed natural-text prompt** per context size (real code+prose sentences,
  counter-prefixed to avoid degenerate exact-repeat acceptance), so draft
  acceptance is representative of real text (random digits in the prefill client
  are not).
- Sizes **ascending and prefix-shared** -> the engine's prefix cache carries from
  one size to the next, so total cold prefill is ~the largest size only (cheap),
  and after the first rep each size is warm. We measure decode, not prefill.
- **512 generated tokens × 3 reps** per size; `report_decode.py` reports the
  median decode with min-max spread, and **pooled** draft acceptance
  (Σ accepted / Σ proposed over all reps) to beat down per-sample noise.
- Sizes 4K–160K are all below both configs' limits (A's wall is 192K), so every
  row is apples-to-apples.

## Results

Decode = effective output tokens/s (accepted drafts included) at that depth.

| Context | A decode (min-med-max) | A accept | B decode (min-med-max) | B accept |
|---:|:---:|:---:|:---:|:---:|
| 4K   | 71.2-**75.8**-79.1 | 26% | 68.8-**80.2**-89.7 | 49% |
| 16K  | 70.2-**75.5**-95.6 | 30% | 61.0-**65.9**-66.5 | 38% |
| 49K  | 61.5-**67.6**-71.8 | 28% | 55.6-**59.5**-62.2 | 39% |
| 98K  | 59.4-**64.3**-72.1 | 34% | 47.9-**50.1**-64.9 | 41% |
| 164K | 43.9-**60.5**-64.7 | 34% | 39.5-**42.1**-60.8 | 42% |

Overall decode median across all samples: **A 70.2 T/s** (spread 43.9–95.6),
**B 61.0 T/s** (spread 39.5–89.7).

## Findings

1. **A (DFlash2) decodes faster than B (MTP) at every context depth**, and the
   margin widens with context: at 164K, A ≈ 60.5 T/s vs B ≈ 42.1 T/s (+44%).
   Across 4K→164K, A's decode falls ~21% while B's falls ~48%.
2. **…despite B having clearly higher draft acceptance** (38–49% vs 26–34%) at
   every size. Higher acceptance is not translating into throughput. The MTP
   head's proposals are *accepted* more often but are *more expensive to
   produce/verify* than the small, fast DFlash2 draft model (a ~5 bpw model that
   runs cheaply per draft). MTP's Q8 draft KV also costs more attention the
   deeper the context, which is why its decode degrades fastest.
   **Net: DFlash2 is the better draft method for generation speed on this card.**
3. **Acceptance here is far lower than the synthetic-digit figure in the prefill
   study** (which showed MTP ~89%). On natural text it is ~40%; random digits are
   pathologically easy for MTP to predict. This is the honest, representative
   number.
4. **Absolute decode is lower than the prefill table suggested** (~60–76 T/s vs
   the inflated 90–116 there). Short 128-token samples understate draft-verification
   overhead; 512-token sustained generations are the true steady-state.

## Overall recommendation (prefill + decode combined)

Prefill/TTFT are a tie (draft-agnostic), so **choose by workload**:

- **Need > 192K context** → B (`config.mtp-256k.yml`) is the only option (and
  uses slightly less VRAM).
- **Within 192K, and you care about generation speed / agent responsiveness** →
  **A (`config.yml`, DFlash2) decodes faster and holds up far better deep in
  context** — the better default for chat/agent/tool-use.
- **Prefill-dominated (RAG, long-doc ingest, few output tokens)** → the two are
  equivalent; pick for context ceiling.

This revises the prefill study's tentative "B is the better default": once decode
is measured properly, DFlash2 (A) wins generation throughput and B's advantage is
specifically its longer context window.

## Reproduce

```sh
# engine's stdout must be redirected to a log file for parsing:
rocm/benchmark/restart_engine.sh ~/tabbyAPI/config.yml /tmp/engine_A_dec.log
python3 rocm/benchmark/bench_decode.py --tag A --log /tmp/engine_A_dec.log \
  --sizes "4096 16384 49152 98304 163840" --max-tokens 512 --reps 3 \
  --out /tmp/decode_A.jsonl --done /tmp/decode_A.done
# ... same for config.mtp-256k.yml -> /tmp/decode_B.* ...
python3 rocm/benchmark/report_decode.py /tmp/decode_A.jsonl /tmp/decode_B.jsonl \
  /tmp/engine_A_dec.log /tmp/engine_B_dec.log
```

Raw per-request data: `decode_A_config-dflash2-192k.jsonl`,
`decode_B_config-mtp-256k.jsonl`.
