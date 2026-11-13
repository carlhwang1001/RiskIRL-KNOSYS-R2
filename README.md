# Immediate-Risk Inverse Reinforcement Learning from Identity-Unlabeled Heterogeneous Demonstrations

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24182602.svg)](https://doi.org/10.5281/zenodo.24182602)

Code and results for the manuscript by Kao-Shing Hwang and Jin-Ling Lin (Knowledge-Based Systems, manuscript KNOSYS-D-26-17431, under review).

Each demonstration trajectory has a latent risk coefficient β under an immediate-risk mean–variance action value; a shared linear reward θ and the posterior over every β are fitted by variational EM with a deterministic log-scale quadrature E-step (K = 64 nodes on [0.02, 10], LogNormal(0, 0.75²) prior) and a finite-difference M-step.

## Layout

| Folder | Contents |
|---|---|
| `exp0/` | Environment, risk-sensitive value iteration, EM core and demonstration generator for the 10×10 stochastic risk grid-world (original code; only the output paths of `run_exp0.py` and `run_chunk.py` were made relative). |
| `experiments/` | Final-protocol runner and all scripts for the revision experiments (see the table below). |
| `results/` | Raw per-run results (JSON Lines) and text summaries used in the manuscript. |
| `dispatch_irl/` | A separate teaching example on production-line dispatching (simulated data; README in Chinese). It is not used for any result in the manuscript. |
| `interactive_demo/` | Static web page that steps through the E-step and M-step on a three-situation teaching model; also hosted at https://stsp-data.com/ai-design/Risk-Sensitive-IRL/index_en.html. |

## Installation

```
pip install -r requirements.txt
```

The results in `results/` were produced with Python 3.13, NumPy 2.5, SciPy 1.18 and Matplotlib 3.11 on a 2-core CPU. Each 100-iteration EM run on the 10×10 grid takes about 35 s on one core.

## Fixed settings and seeds

- True reward θ* = (goal 10, risk −1.6, mild 0.2, step −0.3); γ = 0.95; slip 0.1; rationality λ = 3; R0 = 1.
- Training data: 4 risk profiles β ∈ {0.3, 0.7, 1.5, 2.5} × 20 trajectories, `generate(1234)`.
- Held-out data: 4 × 10 trajectories, seed 5678 (`r2lib.heldout_set`).
- Training seeds 0–9 (initial θ ~ N(0, 0.5²), goal weight projected onto [0, ∞) unless stated).
- Optimizer: Adam, learning rate 0.3, at most 100 EM iterations, stop when ‖Δθ‖ < 1e-3 after iteration 10.
- Screening: a run is removed if its final ELBO is more than 100 nats below the best run of the same condition.

## Reproducing the manuscript

All commands are run from the repository root.

| Manuscript item | Command | Output |
|---|---|---|
| Oracle reference (r = 0.907) | `python experiments/oracle.py` | printed |
| Main table, three aggregation operators (Section 4.4) | `python experiments/final_proto.py vanilla 0 1 2 3 4 5 6 7 8 9` (also `pcgrad`, `cagrad`) | `results/final_runs.jsonl` |
| Shaping invariance check (Proposition 2) | `python experiments/shaping_check.py` | printed |
| Two-step family (Proposition 1, Appendix A) | `python experiments/e5_counterexample.py` | printed (`results/e5_out.txt`) |
| Table 1: within-framework ablations, oracle identity, known β | `python experiments/run_e1e2.py 0 1 results/e1e2.jsonl` then `python experiments/summ_e1e2.py results/e1e2.jsonl` | `results/e1e2.jsonl`, `results/e1e2_summary.txt` |
| Mixture EM, no-projection runs, λ profile, aggregation operators (Sections 4.6–4.8) | `python experiments/run_e367.py 0 1 results/e367.jsonl` then `python experiments/summ_e367.py results/e367.jsonl results/e1e2.jsonl` | `results/e367.jsonl`, `results/e367_summary.txt` |
| Oracle row of Table 1 | `python experiments/oracle_ref.py` | printed |
| Fig. 5 | `python experiments/fig5_data.py` then `python experiments/fig5_draw.py fig5.png` | `results/fig5_data.json`, `fig5.png` |

The two runner scripts accept `<worker_id> <n_workers> <output>` so that the job list can be split across processes (for example, `0 2 out.jsonl` and `1 2 out.jsonl` in parallel). Runners append to their output file; delete it first to start from scratch. A full run of `run_e1e2.py` and `run_e367.py` takes about one hour on two cores.

`experiments/smoke.py` fits one seed of the proposed model (about 35 s) and is a quick installation check.

## Scope

The model penalizes only the conditional variance of the one-step reward; it does not represent return-variance, CVaR or distributional risk (Proposition 1). All experiments are tabular and use simulated demonstrations.

## Citing this code

Please cite the archived release: Kao-Shing Hwang and Jin-Ling Lin, *Immediate-Risk Inverse Reinforcement Learning from Identity-Unlabeled Heterogeneous Demonstrations: code and results* (v1.0), Zenodo, https://doi.org/10.5281/zenodo.24182602. The accompanying article is under review; its reference will be added here after publication.

## Licence

Released under the MIT License; see `LICENSE`.
