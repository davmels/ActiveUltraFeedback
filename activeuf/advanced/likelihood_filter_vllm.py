"""
Compute completion log-likelihoods by launching a vLLM server and querying it.

Usage:
    python activeuf/advanced/likelihood_filter_vllm.py \
        --model-path /path/to/model \
        --dataset-path /path/to/combined_annotated \
        --slurm-nodes 4 --workers 4 --dp-size 4

    # With existing server:
    python activeuf/advanced/likelihood_filter_vllm.py \
        --model-path /path/to/model \
        --dataset-path /path/to/combined_annotated \
        --base-url http://host:8080/v1
"""

import os
import sys
import json
import time
import asyncio
import argparse
import subprocess
import urllib.request
from collections import defaultdict
from multiprocessing import Pool

import httpx
import numpy as np
import uvloop
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer

# ── Globals for multiprocessing (inherited via fork, no pickling) ──
_tokenizer = None
_data = None


def _init_worker(model_path):
    global _tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(model_path)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token


def _tokenize_row(args):
    row_idx, max_length = args
    row = _data[row_idx]
    messages = row["chosen"]
    clean = [{"role": m["role"], "content": m["content"] or ""} for m in messages]

    # Find prompt boundary
    prompt_msgs = clean
    for i in range(len(clean) - 1, -1, -1):
        if clean[i]["role"] == "assistant":
            prompt_msgs = clean[:i]
            break
    if not prompt_msgs:
        return []

    prompt_text = _tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
    prompt_ntokens = len(_tokenizer.encode(prompt_text, add_special_tokens=False))

    annotations = json.loads(row["annotations"]) if isinstance(row["annotations"], str) else row["annotations"]
    results = []
    for ann_idx, ann in enumerate(annotations):
        if not ann.get("response"):
            continue
        full_msgs = prompt_msgs + [{"role": "assistant", "content": ann["response"]}]
        full_text = _tokenizer.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)
        full_ntokens = len(_tokenizer.encode(full_text, add_special_tokens=False))
        if full_ntokens - prompt_ntokens <= 0 or full_ntokens > max_length:
            continue
        results.append({
            "row_idx": row_idx, "ann_idx": ann_idx, "ann_model": ann["model"],
            "full_text": full_text, "comp_start": prompt_ntokens,
        })
    return results


# ── Server launch ──

def launch_server(args):
    scratch = os.environ.get("SCRATCH", "/tmp")
    os.makedirs(args.logs_dir, exist_ok=True)

    fw_args = (
        f"--model {args.model_path} --host 0.0.0.0 --port 8080 "
        f"--served-model-name {args.model_path} "
        f"--data-parallel-size {args.dp_size} --tensor-parallel-size {args.tp_size} "
        f"--trust-remote-code"
    )
    cmd = [
        "python", f"{scratch}/model-launch/serving/submit_job.py",
        "--slurm-nodes", str(args.slurm_nodes), "--slurm-time", args.job_time,
        "--serving-framework", "vllm", "--worker-port", "8080",
        "--slurm-environment", f"{scratch}/model-launch/serving/envs/{args.env}.toml",
        "--framework-args", fw_args,
    ]
    if args.workers > 1:
        cmd += [
            "--workers", str(args.workers), "--nodes-per-worker", str(args.nodes_per_worker),
            "--use-router",
            "--router-environment", f"{scratch}/model-launch/serving/envs/sglang.toml",
        ]

    print(f"Submitting: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=args.logs_dir, capture_output=True, text=True, check=True)

    job_id = None
    for line in (result.stdout + "\n" + result.stderr).splitlines():
        if "Job submitted successfully with ID:" in line:
            job_id = line.split()[-1].strip()
    if not job_id:
        print(result.stdout, result.stderr)
        sys.exit(1)
    print(f"Job ID: {job_id}", flush=True)

    # Wait for URL
    log_file = f"{args.logs_dir}/logs/{job_id}/log.out"
    prefix = "Router URL: " if args.workers > 1 else "All worker URLs: "
    base_url = None
    while not base_url:
        if os.path.exists(log_file):
            with open(log_file) as f:
                for line in f:
                    if line.startswith(prefix):
                        raw = line.split(prefix)[1].strip()
                        base_url = f"{raw}/v1" if args.workers > 1 else f"{raw.rsplit(':', 1)[0]}:8080/v1"
        time.sleep(5)

    # Health check
    health_url = base_url.replace("/v1", "/health")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while True:
        try:
            with opener.open(urllib.request.Request(health_url), timeout=5) as resp:
                if resp.getcode() == 200:
                    break
        except Exception:
            pass
        time.sleep(10)

    print(f"Server ready at {base_url}", flush=True)
    return base_url, job_id


# ── Async scoring ──

async def _writer(queue, path):
    with open(path, "a") as f:
        while True:
            item = await queue.get()
            if item is None:
                break
            f.write(json.dumps(item) + "\n")
            f.flush()
            queue.task_done()


async def _score_batch(batch, client, base_url, model, sem, queue):
    """Score a batch of requests in a single HTTP call."""
    indices, reqs = zip(*batch)
    async with sem:
        try:
            resp = await client.post(f"{base_url}/completions", json={
                "model": model,
                "prompt": [r["full_text"] for r in reqs],
                "max_tokens": 0, "echo": True, "logprobs": 1, "temperature": 0,
            })
            resp.raise_for_status()
            choices = resp.json()["choices"]
            # vLLM returns choices sorted by index field when prompt is a list
            choices.sort(key=lambda c: c.get("index", 0))
            for (idx, req), choice in zip(batch, choices):
                lps = choice["logprobs"]["token_logprobs"]
                comp_lps = [lp for lp in lps[req["comp_start"]:] if lp is not None]
                sum_lp = sum(comp_lps) if comp_lps else None
                n_tok = len(comp_lps)
                await queue.put({
                    "req_idx": idx, "row_idx": req["row_idx"], "ann_idx": req["ann_idx"],
                    "ann_model": req["ann_model"], "sum_logprob": sum_lp, "num_tokens": n_tok,
                })
        except Exception as e:
            print(f"Error batch starting at req {indices[0]}: {e}", flush=True)
            for idx, req in batch:
                await queue.put({
                    "req_idx": idx, "row_idx": req["row_idx"], "ann_idx": req["ann_idx"],
                    "ann_model": req["ann_model"], "sum_logprob": None, "num_tokens": 0,
                })


async def run_scoring(requests, base_url, model, concurrent, jsonl_path, completed,
                      batch_size=64):
    queue = asyncio.Queue()
    writer = asyncio.create_task(_writer(queue, jsonl_path))

    todo = [(i, r) for i, r in enumerate(requests) if i not in completed]
    print(f"Scoring {len(todo)} requests ({len(completed)} already done) "
          f"in batches of {batch_size}...", flush=True)

    # Split into batches
    batches = [todo[i:i + batch_size] for i in range(0, len(todo), batch_size)]

    limits = httpx.Limits(max_connections=concurrent, max_keepalive_connections=concurrent)
    async with httpx.AsyncClient(limits=limits, timeout=httpx.Timeout(7200.0)) as client:
        sem = asyncio.Semaphore(concurrent)
        # Map each future to its batch size for progress tracking
        future_to_size = {}
        for b in batches:
            fut = asyncio.ensure_future(_score_batch(b, client, base_url, model, sem, queue))
            future_to_size[fut] = len(b)
        pbar = tqdm(total=len(todo), desc="Scoring", mininterval=5)
        pending = set(future_to_size.keys())
        while pending:
            done, pending = await asyncio.wait(pending, timeout=2.0, return_when=asyncio.FIRST_COMPLETED)
            pbar.update(sum(future_to_size[f] for f in done))
        pbar.close()

    await queue.put(None)
    await writer


# ── Main ──

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-path", default="likelihood_stats_vllm.json")
    parser.add_argument("--split", default="train")
    parser.add_argument("--concurrent", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--compare-path", default=None)
    parser.add_argument("--output-dataset", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--dp-size", type=int, default=1)
    parser.add_argument("--slurm-nodes", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--nodes-per-worker", type=int, default=1)
    parser.add_argument("--job-time", default="04:00:00")
    parser.add_argument("--logs-dir", default="./logs")
    parser.add_argument("--env", default="vllm")
    args = parser.parse_args()

    # 1. Launch server
    job_id = None
    if args.base_url:
        base_url = args.base_url
    else:
        base_url, job_id = launch_server(args)

    # 2. Load dataset into global (workers inherit via fork)
    global _data
    ds = load_from_disk(args.dataset_path)
    _data = ds[args.split]
    if args.max_samples:
        _data = _data.select(range(min(args.max_samples, len(_data))))

    # 3. Tokenize (parallel)
    print(f"Tokenizing {len(_data)} rows with 288 workers...", flush=True)
    work_items = [(i, args.max_length) for i in range(len(_data))]
    requests = []
    with Pool(288, initializer=_init_worker, initargs=(args.model_path,)) as pool:
        for row_reqs in tqdm(pool.imap(_tokenize_row, work_items, chunksize=64),
                             total=len(_data), desc="Tokenizing", mininterval=5):
            requests.extend(row_reqs)
    print(f"Total requests: {len(requests)}", flush=True)

    # 4. Load checkpoint
    jsonl_path = args.output_path.replace(".json", ".jsonl")
    completed = set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            for line in f:
                if line.strip():
                    e = json.loads(line)
                    if e.get("sum_logprob") is not None:
                        completed.add(e["req_idx"])
        print(f"Resuming: {len(completed)} done", flush=True)

    # 5. Score
    if len(completed) < len(requests):
        uvloop.install()
        asyncio.run(run_scoring(requests, base_url, args.model_path, args.concurrent, jsonl_path, completed,
                               batch_size=args.batch_size))

    # 6. Aggregate
    model_results = defaultdict(list)
    with open(jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue
            e = json.loads(line)
            if e["sum_logprob"] is None:
                continue
            model_results[e["ann_model"]].append({
                "row_idx": e["row_idx"], "sum_logprob": e["sum_logprob"],
                "mean_logprob": e["sum_logprob"] / e["num_tokens"], "num_tokens": e["num_tokens"],
            })

    # 7. Stats
    output = {"evaluator_model": args.model_path, "per_model_stats": {}}
    print("=" * 80)
    for name in sorted(model_results):
        res = model_results[name]
        stats = {k: _stats([r[k] for r in res]) for k in ("mean_logprob", "sum_logprob", "num_tokens")}
        output["per_model_stats"][name] = stats
        print(f"\n{name} (n={len(res)}):")
        print(f"  mean_logprob: mean={stats['mean_logprob']['mean']:.4f} std={stats['mean_logprob']['std']:.4f} "
              f"median={stats['mean_logprob']['median']:.4f}")
        print(f"  sum_logprob:  mean={stats['sum_logprob']['mean']:.2f} std={stats['sum_logprob']['std']:.2f}")

    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output_path}")

    # 8. Comparison
    if args.compare_path:
        with open(args.compare_path) as f:
            hf = json.load(f)["per_model_stats"]
        for name in sorted(set(hf) & set(output["per_model_stats"])):
            h, v = hf[name]["mean_logprob"]["mean"], output["per_model_stats"][name]["mean_logprob"]["mean"]
            print(f"  {name}: HF={h:.6f} vLLM={v:.6f} diff={abs(h-v):.6f}")

    # 9. Save logprobs into dataset
    if args.output_dataset:
        print(f"Saving dataset to {args.output_dataset}...")
        lookup = defaultdict(dict)
        with open(jsonl_path) as f:
            for line in f:
                if not line.strip():
                    continue
                e = json.loads(line)
                if e["sum_logprob"] is not None:
                    lookup[e["row_idx"]][e["ann_idx"]] = e["sum_logprob"]

        new_anns = []
        for i in range(len(_data)):
            row = _data[i]
            anns = json.loads(row["annotations"]) if isinstance(row["annotations"], str) else row["annotations"]
            for j, ann in enumerate(anns):
                if j in lookup.get(i, {}):
                    ann["sum_logprob"] = lookup[i][j]
            new_anns.append(json.dumps(anns) if isinstance(row["annotations"], str) else anns)

        _data.remove_columns("annotations").add_column("annotations", new_anns).save_to_disk(args.output_dataset)
        print(f"Dataset saved to {args.output_dataset}")

    # 10. Cleanup
    if job_id:
        subprocess.run(["scancel", job_id])
        print(f"Cancelled job {job_id}")


def _stats(vals):
    a = np.array(vals)
    return {"count": len(a), "mean": float(a.mean()), "std": float(a.std()),
            "median": float(np.median(a)), "min": float(a.min()), "max": float(a.max()),
            "q25": float(np.percentile(a, 25)), "q75": float(np.percentile(a, 75))}


if __name__ == "__main__":
    main()
