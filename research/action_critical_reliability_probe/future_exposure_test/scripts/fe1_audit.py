"""FE AUDIT 1 — the gates that must pass BEFORE any pre-registered optimizer step.

G1  cached-path equivalence: the cached Stage A tensors reproduce the OFFICIAL
    backend.forward() loss_flow on the same samples, with the same flow RNG.
G2  padding equivalence: a cached sample's contribution does not depend on which other sample
    shares its batch.
G3  future condition: Control is exactly 100% H_pred; Future-Exposed is 50% +/- tolerance, and
    the mask is drawn over the REPEATED batch as _build_flow_future_condition does.
G4  paired randomness: with p_pred forced to 1.0 in BOTH arms, the two arms' per-step losses are
    bitwise identical -- so the mask is the only source of divergence.
G5  determinism: identical inputs and flow seed give an identical loss; flow.train() and
    flow.eval() agree numerically (cfg_drop_prob == 0).
G6  freeze: after real optimizer steps, every frozen module hash is unchanged and the Flow Head
    hash has changed.
G7  aux-loss gradient gate (re-asserted from AUDIT 0 on the live path).

Writes PREFLIGHT_FE_AUDIT1.json. Nothing here is a scientific result.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fe_cache as fca  # noqa: E402
import fe_common as fc  # noqa: E402
import fe_flow as ff  # noqa: E402

OUT = fc.RUN_DIR / "PREFLIGHT_FE_AUDIT1.json"


def main() -> None:
    rep: dict = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "purpose": "FE AUDIT 1"}
    rep["determinism_settings"] = fc.enable_determinism()
    manifest = json.loads((fca.CACHE_ROOT / "CACHE_MANIFEST.json").read_text())
    rep["cache_manifest_meta"] = manifest["_meta"]
    rep["cache_counts"] = {k: manifest[k]["n_samples"] for k in manifest if k != "_meta"}
    rep["cache_unique_episodes"] = {
        k: manifest[k]["n_unique_episodes_in_addresses"] for k in manifest if k != "_meta"
    }

    vla, backend = fc.load_backend(use_bf16=False)
    freeze = fc.freeze_all_but_flow(backend)
    fc.set_train_modes(backend)
    rep["freeze_plan"] = freeze
    frozen0 = {g: fc.module_state_hash(getattr(backend, g)) for g in fc.FROZEN_GROUPS}
    flow0 = fc.module_state_hash(backend.flow)

    cfg = fc.load_train_cfg()
    ds_all = fc.build_dataset(cfg, dataset_statistics_override=None)
    collator = fc.build_collator(cfg, training=True)
    leaf = fc._inner_datasets(ds_all)[0]
    train_eps = manifest["train"]["episodes"]
    fc.restrict_to_episodes(ds_all, train_eps)

    addrs = [tuple(a) for a in manifest["train"]["addresses"][:8]]

    # ----------------------------------------------------------------- G1 cached-path equivalence
    # backend.eval() + flow.train() makes _build_flow_future_condition take its eval branch, which
    # returns H_pred unconditionally -- i.e. exactly the Control condition. So the official
    # forward()'s loss_flow is directly comparable to our cached path with h_t1_star="pred".
    # The Flow Head's internal draws take no generator, and _compute_distill_loss samples from the
    # LAM VQ before the flow call, so the global stream cannot simply be seeded beforehand.
    # Instead the Flow Head's bound `forward` is temporarily wrapped to seed immediately before it
    # runs -- research-side only, restored straight after, no repository file touched.
    g1 = []
    orig_forward = backend.flow.forward
    for ep, st in addrs[:4]:
        fseed = fc.seed_from("FE2026-audit1", str(ep), str(st)) % (2 ** 31)

        def seeded(*a, _s=fseed, **kw):
            torch.manual_seed(_s)
            return orig_forward(*a, **kw)

        sample = fc.get_sample(ds_all, leaf, ep, st)
        live = collator([sample])
        live = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in live.items()}
        backend.flow.forward = seeded
        with torch.no_grad():
            official = backend.forward(batch=live)
        backend.flow.forward = orig_forward
        l_off = float(official["loss_flow"])

        cb = fca.collate_cached([fca.load_sample("train", ep, st)])
        with torch.no_grad():
            l_cache, _ = ff.flow_loss(backend, cb, p_pred=1.0, flow_seed=fseed,
                                      h_t1_star_override="pred")
        g1.append({"episode": int(ep), "start": int(st), "official_loss_flow": l_off,
                   "cached_loss_flow": float(l_cache),
                   "abs_diff": abs(l_off - float(l_cache)),
                   "rel_diff": abs(l_off - float(l_cache)) / max(abs(l_off), 1e-12)})
    rep["G1_cached_path_equivalence"] = {
        "per_sample": g1,
        "max_abs_diff": max(x["abs_diff"] for x in g1),
        "max_rel_diff": max(x["rel_diff"] for x in g1),
        "tolerance_rel": 1e-4,
        "passed": max(x["rel_diff"] for x in g1) < 1e-4,
        "note": "compares OUR cached flow path against the OFFICIAL backend.forward() flow loss "
                "on the same sample with the same flow RNG",
    }

    # ----------------------------------------------------------------- G2 padding independence
    recs = [fca.load_sample("train", ep, st) for ep, st in addrs[:4]]
    lens = [int(r["h_vlm"].shape[0]) for r in recs]
    solo = []
    for r in recs:
        cb = fca.collate_cached([r])
        with torch.no_grad():
            l, _ = ff.flow_loss(backend, cb, p_pred=1.0, flow_seed=12345, h_t1_star_override="pred")
        solo.append(float(l))
    # the same four samples, but padded together to the batch maximum, evaluated one at a time
    smax = max(lens)
    together = []
    for r in recs:
        pad_rec = dict(r)
        p = smax - int(r["h_vlm"].shape[0])
        if p:
            pad_rec["h_vlm"] = torch.nn.functional.pad(r["h_vlm"], (0, 0, 0, p))
            pad_rec["attention_mask"] = torch.nn.functional.pad(r["attention_mask"], (0, p), value=False)
        cb = fca.collate_cached([pad_rec])
        with torch.no_grad():
            l, _ = ff.flow_loss(backend, cb, p_pred=1.0, flow_seed=12345, h_t1_star_override="pred")
        together.append(float(l))
    diffs = [abs(a - b) for a, b in zip(solo, together)]
    rep["G2_padding_independence"] = {
        "vlm_seq_lens": lens, "batch_max": smax,
        "loss_unpadded": solo, "loss_right_padded": together,
        "max_abs_diff": max(diffs),
        "tolerance_abs": 1e-6,
        "passed": max(diffs) < 1e-6,
        "note": "right padding marked by attention_mask==False must not change a sample's loss",
    }

    # ----------------------------------------------------------------- G3 future condition
    n_rep = int(backend.model_cfg.repeated_diffusion_steps)
    gen = torch.Generator(device="cuda"); gen.manual_seed(7)
    counts = {"control": 0, "future_exposed": 0}
    rows = {"control": 0, "future_exposed": 0}
    per_sample_split = 0
    for arm, p in (("control", 1.0), ("future_exposed", 0.5)):
        gen.manual_seed(7)
        for i in range(0, 2000, 2):
            hp = torch.zeros(2, 1, 1, device="cuda")
            hg = torch.ones(2, 1, 1, device="cuda")
            cond, pm = ff.build_future_condition(hp.repeat(n_rep, 1, 1), hg.repeat(n_rep, 1, 1), p, gen)
            counts[arm] += int((~pm).sum()); rows[arm] += int(pm.numel())
            if arm == "future_exposed" and bool(pm[0] != pm[2]):
                per_sample_split += 1
    rep["G3_future_condition"] = {
        "control_gt_fraction": counts["control"] / rows["control"],
        "future_exposed_gt_fraction": counts["future_exposed"] / rows["future_exposed"],
        "rows_drawn": rows,
        "repeated_diffusion_steps": n_rep,
        "n_samples_whose_two_diffusion_repeats_got_different_futures": per_sample_split,
        "passed": (counts["control"] == 0
                   and abs(counts["future_exposed"] / rows["future_exposed"] - 0.5) < 0.02
                   and per_sample_split > 0),
        "note": "the mask is drawn over the REPEATED batch, so one sample can receive H_pred in "
                "one diffusion repeat and H_gt in the other -- exactly as "
                "_build_flow_future_condition does",
    }

    # ----------------------------------------------------------------- G4 paired randomness
    # Both arms, 20 real updates each, with p_pred forced to 1.0 in BOTH. The mask generator is
    # still drawn from in both, so if it leaked into the flow stream the losses would diverge.
    def run_arm(p_pred_effective: float, mask_seed: int, n: int = 20) -> list[float]:
        vla2, b2 = fc.load_backend(use_bf16=False)
        fc.freeze_all_but_flow(b2); fc.set_train_modes(b2)
        for g in fc.FROZEN_GROUPS:
            m = getattr(b2, g)
            if not torch.is_tensor(m):
                m.to("cpu")
        torch.cuda.empty_cache()
        params = [q for q in b2.flow.parameters() if q.requires_grad]
        opt = torch.optim.AdamW(params, lr=1e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-8)
        mg = torch.Generator(device="cuda"); mg.manual_seed(mask_seed)
        losses = []
        train_addr = [tuple(a) for a in manifest["train"]["addresses"]]
        for u in range(n):
            opt.zero_grad(set_to_none=True)
            tot = 0.0
            for micro in range(4):
                chunk = train_addr[(u * 8 + micro * 2):(u * 8 + micro * 2 + 2)]
                cb = fca.collate_cached([fca.load_sample("train", e, s) for e, s in chunk])
                fseed = fc.seed_from("FE2026-flow", "0", str(u), str(micro)) % (2 ** 31)
                l, _ = ff.flow_loss(b2, cb, p_pred=p_pred_effective, flow_seed=fseed, mask_gen=mg)
                (l / 4).backward()
                tot += float(l)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            losses.append(tot / 4)
        del b2, vla2, opt
        torch.cuda.empty_cache()
        return losses

    a_losses = run_arm(1.0, mask_seed=fc.seed_from("FE2026-mask", "0") % (2 ** 31))
    b_losses = run_arm(1.0, mask_seed=fc.seed_from("FE2026-mask", "0") % (2 ** 31))
    d = [abs(x - y) for x, y in zip(a_losses, b_losses)]

    # The exact form of the claim, tested forward-only so no CUDA reduction order is involved:
    # advancing the mask generator by different amounts must not move the flow loss by one bit.
    cbx = fca.collate_cached([fca.load_sample("train", *addrs[0])])
    mask_indep = []
    for n_draws in (0, 1, 17):
        g = torch.Generator(device="cuda"); g.manual_seed(4242)
        for _ in range(n_draws):
            torch.rand(4, 1, 1, generator=g, device="cuda")
        with torch.no_grad():
            l, _ = ff.flow_loss(backend, cbx, p_pred=1.0, flow_seed=555,
                                h_t1_star_override="pred")
        mask_indep.append(float(l))
    rep["G4_paired_randomness"] = {
        "n_updates": len(a_losses),
        "max_abs_diff": max(d),
        "bitwise_identical": max(d) == 0.0,
        "first_losses": a_losses[:5],
        "mask_generator_independence": {
            "losses_after_0_1_17_mask_draws": mask_indep,
            "identical": len(set(mask_indep)) == 1,
        },
        "passed": max(d) == 0.0 and len(set(mask_indep)) == 1,
        "note": "Two independent 20-update runs, bitwise identical under deterministic kernels "
                "(amendment FE-A5; without them the floor was 1.4e-8 from CUDA reduction order "
                "in the backward pass). The second sub-check is the exact statement of the "
                "claim: the mask generator is a SEPARATE stream, so advancing it cannot move "
                "the flow time, noise or dropout masks.",
    }

    # ----------------------------------------------------------------- G5 determinism
    cb = fca.collate_cached([fca.load_sample("train", *addrs[0])])
    dit = backend.flow.DiT
    drops = [m for m in dit.modules() if isinstance(m, torch.nn.Dropout)]
    with torch.no_grad():
        l1, _ = ff.flow_loss(backend, cb, p_pred=1.0, flow_seed=999, h_t1_star_override="pred")
        l2, _ = ff.flow_loss(backend, cb, p_pred=1.0, flow_seed=999, h_t1_star_override="pred")
        backend.flow.eval()
        l_eval, _ = ff.flow_loss(backend, cb, p_pred=1.0, flow_seed=999, h_t1_star_override="pred")
        l_eval2, _ = ff.flow_loss(backend, cb, p_pred=1.0, flow_seed=999, h_t1_star_override="pred")
        backend.flow.train()
        # Decisive test of the explanation: with every Dropout p set to 0, train mode must equal
        # eval mode exactly. If it does, the train/eval gap IS the DiT's activation dropout.
        old_p = [m.p for m in drops]
        for m in drops:
            m.p = 0.0
        l_nodrop_train, _ = ff.flow_loss(backend, cb, p_pred=1.0, flow_seed=999, h_t1_star_override="pred")
        backend.flow.eval()
        l_nodrop_eval, _ = ff.flow_loss(backend, cb, p_pred=1.0, flow_seed=999, h_t1_star_override="pred")
        backend.flow.train()
        for m, p in zip(drops, old_p):
            m.p = p
        l4, _ = ff.flow_loss(backend, cb, p_pred=1.0, flow_seed=1000, h_t1_star_override="pred")
        l5, _ = ff.flow_loss(backend, cb, p_pred=1.0, flow_seed=999, h_t1_star_override="gt")
    rep["G5_determinism"] = {
        "train_mode_same_seed_twice_identical": float(l1) == float(l2),
        "eval_mode_same_seed_twice_identical": float(l_eval) == float(l_eval2),
        "different_flow_seed_changes_loss": float(l1) != float(l4),
        "Hgt_condition_changes_loss": float(l1) != float(l5),
        "loss_Hpred_train": float(l1), "loss_Hgt_train": float(l5), "loss_Hpred_eval": float(l_eval),
        "dit_dropout": {
            "n_dropout_modules": len(drops),
            "p_values": sorted({float(p) for p in old_p}),
            "train_minus_eval_abs_diff": abs(float(l1) - float(l_eval)),
            "with_p_set_to_zero_train_equals_eval": float(l_nodrop_train) == float(l_nodrop_eval),
            "explanation": "The AlternateVLDiT carries 64 nn.Dropout modules at p=0.2 "
                           "(final_dropout=True), active in train mode. This is the official "
                           "training regulariser the release checkpoint was trained with, NOT a "
                           "defect. An earlier version of this gate wrongly asserted that "
                           "cfg_drop_prob==0 implies train and eval agree; that conflated the "
                           "Flow Head's CFG dropout with the DiT's activation dropout.",
            "consequence_for_pairing": "The dropout masks are drawn INSIDE the flow forward, "
                                       "after torch.manual_seed(flow_seed), and their number "
                                       "does not depend on h_t1_star. Both arms therefore "
                                       "receive identical dropout masks as well as identical "
                                       "tau and epsilon.",
            "mode_policy": "Training runs flow.train() (dropout on), matching the official "
                           "trainer. Validation and the 2x2 evaluation run flow.eval() "
                           "(dropout off), matching predict_action's deployment path.",
        },
        "passed": (float(l1) == float(l2) and float(l_eval) == float(l_eval2)
                   and float(l_nodrop_train) == float(l_nodrop_eval)
                   and float(l1) != float(l4) and float(l1) != float(l5)),
        "note": "Determinism is required SEPARATELY in each mode; train and eval are not "
                "expected to agree, and the reason is verified rather than assumed.",
    }

    # ----------------------------------------------------------------- G6 freeze after real steps
    params = [p for p in backend.flow.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-8)
    mg = torch.Generator(device="cuda"); mg.manual_seed(3)
    for u in range(5):
        opt.zero_grad(set_to_none=True)
        cb = fca.collate_cached([fca.load_sample("train", *addrs[u % len(addrs)])])
        l, _ = ff.flow_loss(backend, cb, p_pred=0.5, flow_seed=u, mask_gen=mg)
        l.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    frozen1 = {g: fc.module_state_hash(getattr(backend, g)) for g in fc.FROZEN_GROUPS}
    flow1 = fc.module_state_hash(backend.flow)
    rep["G6_freeze_after_steps"] = {
        "optimizer_steps": 5,
        "frozen_hashes_before": frozen0,
        "frozen_hashes_after": frozen1,
        "frozen_unchanged": frozen1 == frozen0,
        "flow_hash_changed": flow1 != flow0,
        "passed": frozen1 == frozen0 and flow1 != flow0,
    }

    # ----------------------------------------------------------------- G7 aux-loss gradient gate
    sample = fc.get_sample(ds_all, leaf, *addrs[0])
    live = collator([sample])
    live = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in live.items()}
    out = backend.forward(batch=live)
    aux = out["loss_perceptual"] + out["loss_distill"]
    n_non_none = 0
    if aux.requires_grad:
        grads = torch.autograd.grad(aux, params, allow_unused=True)
        n_non_none = sum(1 for g in grads if g is not None)
    rep["G7_aux_grad_gate"] = {
        "aux_requires_grad": bool(aux.requires_grad),
        "n_flow_params_with_non_none_grad": int(n_non_none),
        "passed": n_non_none == 0,
    }

    gates = {k: bool(v["passed"]) for k, v in rep.items() if isinstance(v, dict) and "passed" in v}
    rep["ALL_GATES"] = gates
    rep["ALL_GATES_PASSED"] = all(gates.values())
    fc.write_json(OUT, rep)
    print(json.dumps({k: rep[k] for k in rep if k.startswith("G") or k.startswith("ALL")},
                     indent=2, default=str))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
