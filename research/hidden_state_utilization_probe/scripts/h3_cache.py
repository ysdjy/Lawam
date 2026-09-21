"""Build the frozen-encoding cache for every teacher sample (model side, `lawam` env).

For each sample the shared encoding is computed exactly as `predict_action` does
(`backend._run_shared_encoding_infer`) and stored: h_t, H_pred, h_vlm, attention_mask. The VLM /
LAM / LaWM stages are frozen and in eval mode, so the cache is exact, not an approximation, and the
adapter training below never has to re-run the 2B VLM.

Usage: python h3_cache.py <run_id>
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hsu_model as hm  # noqa: E402

RUN = sys.argv[1]
OUT = hm.REPO / "results" / "hidden_state_utilization_probe" / RUN
EP = OUT / "episodes"
CACHE = OUT / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
EMBODIMENT_ID, ACTION_HZ = 25, 20.0


def main() -> None:
    stats = hm.state_norm_stats()
    vla, backend = hm.load_backend(use_bf16=True)          # the deployment cast, as in every round
    backend.eval()
    builder = vla.policy_runner.infer_batch_builder

    files = sorted(EP.glob("*.npz"))
    t0, n_total = time.time(), 0
    for f in files:
        out = CACHE / f.name
        if out.exists():
            continue
        d = np.load(f)
        meta = json.loads((EP / f"{f.stem}.json").read_text())
        h_t, h_pred, h_vlm, attn = [], [], [], []
        for i in range(d["primary"].shape[0]):
            example = {
                "primary_image": [np.ascontiguousarray(d["primary"][i])],
                "wrist_image": [np.ascontiguousarray(d["wrist"][i])],
                "lang": meta["lang"],
                "state": hm.normalize_state(d["state"][i], stats),
                "embodiment_id": EMBODIMENT_ID,
                "action_hz": ACTION_HZ,
            }
            batch = builder.build_infer_batch([example])
            with torch.inference_mode():
                shared = backend._run_shared_encoding_infer(
                    prepared_batch=batch, source="hsu.h3_cache", lam_features_with_no_grad=False)
            h_t.append(shared.h_t[0].float().cpu().numpy().astype(np.float16))
            h_pred.append(shared.h_t1_pred[0].float().cpu().numpy().astype(np.float16))
            h_vlm.append(shared.h_vlm[0].float().cpu().numpy().astype(np.float16))
            attn.append((batch["attention_mask"][0] == 1).cpu().numpy())
        lens = {a.shape[0] for a in h_vlm}
        np.savez(out, h_t=np.stack(h_t), h_pred=np.stack(h_pred), h_vlm=np.stack(h_vlm),
                 attn=np.stack(attn), norm_chunk=d["norm_chunk"], env_chunk=d["env_chunk"],
                 step=d["step"], c=np.full(len(h_t), meta["c"], np.float32),
                 mu_eff=np.full(len(h_t), meta["mu_eff"], np.float32))
        n_total += len(h_t)
        print(f"{f.stem}: {len(h_t)} samples, vlm_len={sorted(lens)}, "
              f"{time.time()-t0:.0f}s total", flush=True)
    print(json.dumps({"episodes": len(files), "new_samples": n_total,
                      "cache_dir": str(CACHE), "seconds": round(time.time() - t0, 1)}, indent=2))


if __name__ == "__main__":
    main()
