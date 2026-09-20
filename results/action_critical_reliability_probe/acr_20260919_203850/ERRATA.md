# ERRATA — run `acr_20260919_203850`

Added 2026-09-20, during the G2 audit of this run. **`REPORT.md` itself is deliberately left unmodified**,
so that the record of what was originally written is preserved; this file lists the corrections.

## Five figures in `REPORT.md` §7 (U–S Complementarity) were taken from an intermediate analysis

While preparing the round, `a2_complementarity.py` was run twice on partial data (56 and 59 of the 90
states) to smoke-test the analysis code, and once more after all 90 states were complete. Section §7 of
`REPORT.md` was written from the **59-state** printout; every other section (§5, §6, §8, §9) was written
from the final 90-state products.

| `REPORT.md` §7 as written | correct value (90 states, `summary.json`) |
|---|---|
| Spearman(U, S) median **0.024** | **0.053** |
| Spearman(U, S) p25 **−0.035** | **−0.015** |
| Spearman(U, S) p75 **0.091** | **0.095** |
| Pearson(U, S) **0.045** | **0.037** |
| low-U ∧ high-S **8.2 %** | **7.4 %** |
| S vs token-norm ρ ≈ **−0.32** | **−0.348** |

Unchanged and correct as printed in §7: top-quartile overlap **1.25×** chance, high-U ∧ high-S **7.8 %**,
high-U ∧ low-S **4.7 %**, U vs token-norm ρ **−0.55**, Spearman(U,S) range **−0.20 … 0.19**, and the
statement that the redundancy criterion |ρ| ≥ 0.8 is met in 0 of 90 states.

## Does any conclusion change?

**No.** The claim §7 supports is that U and S are empirically independent within a state. A median Spearman
of 0.053 is as far from the pre-registered redundancy threshold of 0.8 as 0.024 is, the four-quadrant
populations are essentially unchanged, and the Case-C trigger (`|ρ| ≥ 0.8`) remains untripped in every one
of the 90 states. §11's "For" bullet ("U and S are empirically independent (ρ ≈ 0.02)") should read
**ρ ≈ 0.05**; §12's Final Decision is unaffected.

## Authoritative source

For any downstream use, **`summary.json` is authoritative**, together with `metrics.csv`,
`metrics_step34.csv` and the raw `*_records.jsonl` — all of which are the final 90-state products and are
internally consistent. The G2 round used `summary.json` throughout and did not rely on the §7 figures.

Detection and verification: `results/action_critical_reliability_probe/acr_learned_u_20260920_120110/AUDIT_G2.md`,
section "G2 AUDIT 0 §3", where 11 headline figures of `REPORT.md` were cross-checked against `summary.json`
(8 agreed exactly, these 5 did not).
