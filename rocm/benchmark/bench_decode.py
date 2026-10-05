#!/usr/bin/env python3
"""Decode-focused long-context benchmark for TabbyAPI (exllamav3).

Measures tokens/sec and speculative-draft acceptance at a range of context
DEPTHS. Complements bench_longctx.py, which uses cold/unique prompts to measure
prefill. Here the opposite is wanted: a FIXED natural-text prompt per size,
repeated across reps, so

  * after the first request of a size it is KV-cache warm (cheap; we don't care
    about prefill here, only steady-state decode at that context depth), and
  * sizes are ASCENDING and share a prefix, so the engine's prefix cache carries
    from one size to the next -> total cold prefill is ~the largest size only.

Long 512-token generations over real code/prose give stable decode_tps and a
representative draft-acceptance rate (random digits in bench_longctx.py do not).

Reuse from bench_longctx: config (env), post(), read_engine(), max_req(),
wait_up(), read_key(). Run from the host serving the engine (localhost).
"""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_longctx as B

# natural seed lines (code + prose), cycled with a counter so it stays
# human-language and avoids degenerate exact-repeat draft super-acceptance.
SEED = [
    "The optimization pass walks the abstract syntax tree and rewrites dead branches so the emitted kernel keeps only the live data paths for this specialization.",
    "In practice the key-value cache is allocated up front, which means memory use stays flat with respect to how much of the context window a request actually fills.",
    "def apply_patch(path, old, new):",
    "    text = open(path).read()",
    "    assert old in text, f'anchor missing in {path}'",
    "    open(path, 'w').write(text.replace(old, new, 1))",
    "    return True",
    "Speculative decoding drafts several tokens at once and the target model accepts the longest agreeing prefix, so throughput rises with acceptance rate up to the point where verifying drafts costs more than it saves.",
    "Long-context decoding on a hybrid model stays cheap because most layers carry a constant-size recurrent state instead of a cache that grows with the sequence.",
]
TAIL = "\n\nPlease continue writing the next section in detailed, specific prose for several hundred words."


def build_prefix(n_tokens, ratio):
    """Deterministic natural text of ~n_tokens; ascending sizes share this prefix."""
    target_chars = int(n_tokens * ratio)
    out, n, i = [], 0, 0
    while n < target_chars:
        line = f"{i}. {SEED[i % len(SEED)]}"
        out.append(line); n += len(line) + 1; i += 1
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--sizes", required=True, help="ascending target context tokens")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--done", required=True)
    a = ap.parse_args()
    key = B.read_key()
    if not B.wait_up(key):
        open(a.done, "w").write("server-not-up\n"); raise SystemExit("server not up")
    if not os.path.exists(a.log):
        open(a.log, "a").close()
    req_max = B.max_req(a.log)

    # calibrate chars/token for natural text with one tiny request
    cal = build_prefix(16_000, 4.0)
    _, _, _, cerr = B.post(cal + TAIL, 1, key)
    if cerr:
        open(a.done, "w").write("calib-fail: " + cerr + "\n"); raise SystemExit("calib failed: " + cerr)
    eng = B.read_engine(a.log, req_max)
    if not eng or not eng["prompt"]:
        open(a.done, "w").write("calib-unparsed\n"); raise SystemExit("calib unparsed")
    req_max = eng["req"]
    ratio = len(cal) / eng["prompt"]
    print(f"# calib: {len(cal)} chars = {eng['prompt']} tok -> {ratio:.3f} chars/token", flush=True)

    sizes = [int(x) for x in a.sizes.split()]
    out = open(a.out, "a", buffering=1)
    for sz in sizes:
        base = build_prefix(sz, ratio)          # deterministic; shared across sizes/reps
        for rep in range(a.reps):
            prompt = base + TAIL
            wall, ttft, nchunk, err = B.post(prompt, a.max_tokens, key)
            row = {"tag": a.tag, "size": sz, "rep": rep, "chars": len(prompt),
                   "client_ttft_s": round(ttft, 2), "client_wall_s": round(wall, 2)}
            if err:
                row["error"] = err
                out.write(json.dumps(row) + "\n"); print(json.dumps(row), flush=True); continue
            e = B.read_engine(a.log, req_max)
            if e:
                req_max = e["req"]; row.update(e)
            else:
                row["error"] = "log-line-not-captured"
            out.write(json.dumps(row) + "\n"); print(json.dumps(row), flush=True)
            time.sleep(0.5)
    out.close()
    open(a.done, "w").write("ok\n")


if __name__ == "__main__":
    main()
