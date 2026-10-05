#!/usr/bin/env python3
"""Compare two long-context bench runs (A vs B). Optionally recover draft-acceptance
by request number from the engine stdout logs (tolerant/comma-safe), so rows whose
inline draft parse failed are still reported. Reads files only; never contacts the engine.

usage: report.py benchA.jsonl benchB.jsonl [engineA.log engineB.log]
"""
import json, re, sys

DRAFT_RE = re.compile(
    r"#(?P<req>\d+).*?tokens generated at"
    r".*?draft (?P<acc>[\d,]+)/(?P<den>[\d,]+) accepted(?: \((?P<pct>\d+)%\))?"
)

def _i(x):
    return int(x.replace(",", "")) if x else None

_TS = re.compile(r"\d\d:\d\d:\d\d\.\d+\b")

def logical_lines(txt):
    """Rejoin records that the rich logger wrapped across multiple physical lines.
    A new record starts at a timestamped line; other lines are continuations."""
    out = []
    for raw in txt.splitlines():
        if _TS.match(raw):
            out.append(raw.rstrip())
        elif out:
            out[-1] += " " + raw.strip()
        elif raw.strip():
            out.append(raw.strip())
    return [l for l in out if l]

def recover_draft(log):
    m = {}
    try:
        txt = open(log, encoding="utf-8", errors="replace").read()
    except OSError:
        return m
    for line in logical_lines(txt):
        if "tokens generated at" not in line:
            continue
        g = DRAFT_RE.search(line)
        if g:
            acc, den, pct = _i(g["acc"]), _i(g["den"]), _i(g["pct"])
            if pct is None and acc is not None and den:
                pct = round(100.0 * acc / den)
            m[_i(g["req"])] = {"acc": acc, "den": den, "pct": pct}
    return m

def load(path, draftmap):
    rows, errs = {}, {}
    try:
        for l in open(path):
            l = l.strip()
            if not l:
                continue
            r = json.loads(l)
            if r.get("error"):
                errs[r["target"]] = r["error"]
            elif r.get("req") in draftmap:      # authoritative draft from log
                r["draft_pct"] = draftmap[r["req"]]["pct"]
                a, d = draftmap[r["req"]]["acc"], draftmap[r["req"]]["den"]
                r["draft_ratio"] = round(a / d, 3) if (a is not None and d) else None
                r["draft_den"] = d
                rows[r["target"]] = r
            else:
                rows[r["target"]] = r
    except FileNotFoundError:
        pass
    return rows, errs

def cell(r, err):
    if err:
        return "        over-context (400)"
    if not r:
        return "                              -"
    dp = f"{r['draft_pct']}%" if r.get("draft_pct") is not None else "-"
    return (f"{r['prompt']:>7} {r['prefill_tps']:>7} {r['ttft_s']:>6}s "
            f"{r['decode_tps']:>5} {dp:>5}")

def main():
    fa, fb = sys.argv[1], sys.argv[2]
    dA = recover_draft(sys.argv[3]) if len(sys.argv) > 3 else {}
    dB = recover_draft(sys.argv[4]) if len(sys.argv) > 4 else {}
    A, Ae = load(fa, dA); B, Be = load(fb, dB)
    targets = sorted(set(list(A) + list(B) + list(Ae) + list(Be)))
    print("A = running config.yml (DFlash2 draft model, 192K)   |   "
          "B = config.mtp-256k.yml (MTP head draft, 256K)")
    col = " prompt  prefill  ttft  decode draft"
    print(f"{'target':>7} |{col} A |{col} B")
    print("-" * (7 + 3 + 30 + 3 + 30))
    for t in targets:
        print(f"{t:>7} | {cell(A.get(t), Ae.get(t))} | {cell(B.get(t), Be.get(t))}")
    print("\nA prefill T/s (cold):", {t: A[t]['prefill_tps'] for t in sorted(A)})
    print("B prefill T/s (cold):", {t: B[t]['prefill_tps'] for t in sorted(B)})

if __name__ == "__main__":
    main()
