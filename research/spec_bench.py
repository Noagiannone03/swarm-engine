"""Single-GPU speculative-decoding benchmark (validation harness).

Validates the speculative-decoding CORE on REAL models before the distributed
integration: a small same-family DRAFT proposes K tokens, the full TARGET
verifies them in ONE forward pass, and we accept the longest greedy-matching
prefix (parallax.server.speculative.greedy_accept). Measures:

  * correctness: spec output is token-for-token identical to plain greedy
    decode (the oracle) — speculation must never change the result;
  * acceptance: mean tokens accepted per verify round → this is the number that
    predicts the distributed WAN speed-up (≈ accepted+1 tokens per pipeline
    round-trip instead of 1);
  * local speed-up: tok/s spec vs greedy on one GPU (a lower bound; the WAN win
    is larger because there each round-trip is far more expensive).

This pumps Shard's loop (phase0/specdec.py) but with a WHOLE target on one GPU
(no socket split), since the two RunPod pods can't open a direct socket
(NAT-isolated). The acceptance rate it measures is network-independent, so it
directly forecasts the 2-pod WAN result.

Run on a pod:
  python research/spec_bench.py --target Qwen/Qwen3-8B --draft Qwen/Qwen3-0.6B --K 6 --max-new 160
"""

import argparse
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

sys.path.insert(0, "src")
from parallax.server.speculative import greedy_accept  # noqa: E402

CODE_PROMPTS = [
    "Write a Python function quicksort(arr) that sorts a list. Code only.",
    "Implement a Python LRU cache class with get(key) and put(key, value). Code only.",
    "Write a Python function that reads a JSON file and returns the parsed dict, with error handling.",
]


@torch.no_grad()
def greedy_baseline(model, tok, prompt, max_new, dev):
    ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, return_tensors="pt",
    ).to(dev)
    cache = DynamicCache()
    out = model(input_ids=ids, past_key_values=cache, use_cache=True)
    cur = int(out.logits[0, -1].argmax())
    toks = [cur]
    t0 = time.time()
    for _ in range(max_new - 1):
        if cur == tok.eos_token_id:
            break
        out = model(input_ids=torch.tensor([[cur]], device=dev),
                    past_key_values=cache, use_cache=True)
        cur = int(out.logits[0, -1].argmax())
        toks.append(cur)
    dt = time.time() - t0
    return toks, len(toks) / max(dt, 1e-9)


@torch.no_grad()
def spec_decode(target, draft, tok, prompt, K, max_new, dev):
    """Shard's greedy spec loop, single-GPU whole target."""
    ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, return_tensors="pt",
    ).to(dev)
    tcache, dcache = DynamicCache(), DynamicCache()
    # prefill both on the prompt
    cur = int(target(input_ids=ids, past_key_values=tcache, use_cache=True).logits[0, -1].argmax())
    draft(input_ids=ids, past_key_values=dcache, use_cache=True)
    pos = ids.shape[1]
    out = [cur]
    rounds = accepted_total = 0
    t0 = time.time()
    while len(out) < max_new and cur != tok.eos_token_id:
        # 1. draft proposes K tokens (feed cur, keep d_1..d_K)
        drafts, dtok = [], cur
        for i in range(K + 1):
            dl = draft(input_ids=torch.tensor([[dtok]], device=dev),
                       past_key_values=dcache, use_cache=True).logits
            dtok = int(dl[0, -1].argmax())
            if i < K:
                drafts.append(dtok)
        # 2. target verifies [cur, d_1..d_K] in ONE forward
        seq = torch.tensor([[cur] + drafts], device=dev)
        logits = target(input_ids=seq, past_key_values=tcache, use_cache=True).logits
        target_argmax = logits[0].argmax(-1).tolist()  # K+1 predictions
        # 3. greedy acceptance (our module)
        committed, n = greedy_accept(drafts, target_argmax)
        out.extend(committed)
        cur = committed[-1]
        pos += len(committed)
        rounds += 1
        accepted_total += n
        # 4. crop both caches back to the committed length (drop rejected tail)
        tcache.crop(pos)
        dcache.crop(pos)
        if tok.eos_token_id in committed:
            break
    dt = time.time() - t0
    if tok.eos_token_id in out:
        out = out[: out.index(tok.eos_token_id)]
    return out, {
        "tok_s": len(out) / max(dt, 1e-9),
        "rounds": rounds,
        "mean_accept": accepted_total / max(rounds, 1),
        "toks_per_round": (accepted_total + rounds) / max(rounds, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--max-new", type=int, default=160)
    args = ap.parse_args()
    dev = "cuda"

    print(f"loading target {args.target} ...", flush=True)
    target = AutoModelForCausalLM.from_pretrained(args.target, dtype=torch.bfloat16,
                                                  attn_implementation="eager").to(dev).eval()
    print(f"loading draft {args.draft} ...", flush=True)
    draft = AutoModelForCausalLM.from_pretrained(args.draft, dtype=torch.bfloat16,
                                                 attn_implementation="eager").to(dev).eval()
    tok = AutoTokenizer.from_pretrained(args.target)
    print(f"gpu_mem={torch.cuda.memory_allocated(dev)/1e9:.1f}GB; K={args.K}\n", flush=True)

    agg_accept, agg_spec_ts, agg_base_ts, identical = [], [], [], 0
    for i, p in enumerate(CODE_PROMPTS):
        base_toks, base_ts = greedy_baseline(target, tok, p, args.max_new, dev)
        spec_toks, m = spec_decode(target, draft, tok, p, args.K, args.max_new, dev)
        ok = spec_toks[: len(base_toks)] == base_toks[: len(spec_toks)]
        identical += int(ok)
        agg_accept.append(m["mean_accept"]); agg_spec_ts.append(m["tok_s"]); agg_base_ts.append(base_ts)
        print(f"[prompt {i+1}] identical_to_greedy={ok} | mean_accept={m['mean_accept']:.2f} "
              f"tokens/round={m['toks_per_round']:.2f} | spec={m['tok_s']:.1f} tok/s "
              f"greedy={base_ts:.1f} tok/s | speedup={m['tok_s']/max(base_ts,1e-9):.2f}x", flush=True)

    n = len(CODE_PROMPTS)
    print(f"\n=== SUMMARY ({n} code prompts, K={args.K}) ===")
    print(f"correctness: {identical}/{n} identical to greedy")
    print(f"mean accepted/round: {sum(agg_accept)/n:.2f}  -> ~{sum(agg_accept)/n + 1:.1f} tokens per verify traversal")
    print(f"local speedup: {(sum(agg_spec_ts)/n)/max(sum(agg_base_ts)/n,1e-9):.2f}x "
          f"(spec {sum(agg_spec_ts)/n:.1f} vs greedy {sum(agg_base_ts)/n:.1f} tok/s)")
    print("NOTE: over a WAN pipeline the speedup is larger — each accepted token "
          "saved is a saved pipeline round-trip, not just saved local compute.")


if __name__ == "__main__":
    main()
