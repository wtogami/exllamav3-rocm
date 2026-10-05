#!/usr/bin/env python3
"""Summarize two decode-benchmark runs (A vs B) by context depth.

For each (tag,size): median decode_tps across reps (with min-max spread) and a
POOLED draft-acceptance = sum(accepted) / sum(proposed) over all reps. Draft
counts are recovered from the engine log by request number (comma-safe, log-line
unwrap), so rows whose inline draft parse missed are still counted.

usage: report_decode.py decodeA.jsonl decodeB.jsonl [engineA.log engineB.log]
"""
import json, os, statistics, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import report as R


def load(path, draftmap):
    by = {}
    if not os.path.exists(path):
        return by
    for l in open(path):
        l = l.strip()
        if not l:
            continue
        r = json.loads(l)
        if r.get("error") or not r.get("gen_tokens"):
            continue
        req = r.get("req")
        if req in draftmap:
            r["draft_acc"] = draftmap[req]["acc"]; r["draft_den"] = draftmap[req]["den"]
        by.setdefault(r["size"], []).append(r)
    return by


def agg(rows):
    dec = [x["decode_tps"] for x in rows if x.get("decode_tps")]
    acc = sum(x["draft_acc"] for x in rows if x.get("draft_acc") is not None)
    den = sum(x["draft_den"] for x in rows if x.get("draft_den") is not None)
    gen = max((x.get("gen_tokens") or 0) for x in rows) if rows else 0
    return {
        "n": len(rows), "gen": gen,
        "dec_med": round(statistics.median(dec), 1) if dec else None,
        "dec_min": round(min(dec), 1) if dec else None,
        "dec_max": round(max(dec), 1) if dec else None,
        "accept": round(100.0 * acc / den) if den else None,
    }


def main():
    fa, fb = sys.argv[1], sys.argv[2]
    dA = R.recover_draft(sys.argv[3]) if len(sys.argv) > 3 else {}
    dB = R.recover_draft(sys.argv[4]) if len(sys.argv) > 4 else {}
    A, B = load(fa, dA), load(fb, dB)
    sizes = sorted(set(A) | set(B))
    print("A = config.yml (DFlash2 draft model, 192K)   |   B = config.mtp-256k.yml (MTP head draft, 256K)")
    print(f"{'context':>8} | {'A decode (min-med-max)':>26} {'acc':>4} | {'B decode (min-med-max)':>26} {'acc':>4}")
    print("-" * 82)
    for s in sizes:
        ca = agg(A[s]) if s in A else None
        cb = agg(B[s]) if s in B else None
        fa_ = f"{ca['dec_min']}-{ca['dec_med']}-{ca['dec_max']}".rjust(26) + f" {ca['accept']:>3}%" if ca else " " * 31
        fb_ = f"{cb['dec_min']}-{cb['dec_med']}-{cb['dec_max']}".rjust(26) + f" {cb['accept']:>3}%" if cb else " " * 31
        print(f"{s:>8} | {fa_} | {fb_}")
    print("\ndecode = effective output tokens/s (accepted drafts included) at that context depth;"
          "\n  median of reps with min-max spread. acc = pooled draft acceptance across reps.")
    # per-rep spread summary
    for tag, D in (("A", A), ("B", B)):
        decs = [x["decode_tps"] for r in D.values() for x in r if x.get("decode_tps")]
        if decs:
            print(f"  {tag}: {len(decs)} gen-samples, decode median {round(statistics.median(decs),1)} "
                  f"T/s, spread {round(min(decs),1)}-{round(max(decs),1)}")


if __name__ == "__main__":
    main()
