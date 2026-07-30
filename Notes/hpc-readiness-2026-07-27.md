# HPC readiness evaluation — RD training campaigns

Date: 2026-07-27
Purpose: assess what it would take to move the phi-field training campaigns (DE over 64 block values, 4 sims per candidate) onto Cranfield HPC (Crescent2, SLURM), and whether it is worth it.

## Workload profile

- One candidate evaluation = 4 independent single-core sims (739 steps, ~130x240 grid after the crop trim), each ~1-4 min, ~100 MB RAM. No GPU, no BLAS dependence to speak of (memory-bandwidth-bound numpy).
- One DE generation = popsize*64 = 256 candidates = 1024 sims — embarrassingly parallel.
- The only serial point is the DE driver between generations (milliseconds).
- Campaign: 5 generations ≈ 5120 sims.

## Fit: excellent

This is the textbook cluster workload: many small single-core jobs, no inter-node communication, tiny data. A single 64-128 core node runs a generation in roughly (1024 sims x 2 min) / 96 cores ≈ 20-30 min; a full 5-gen campaign in 1.5-2.5 h. Compare: ~15 h/generation on the laptop (thermally throttled).

## Recommended architecture (simplest that works)

- ONE SLURM job, one node, all cores: the existing script already parallelises a generation over a process pool — on the cluster we just set the pool size to the node core count. No MPI, no array-job bookkeeping, no code changes beyond a `--workers` CLI flag and `--poolsize` already present.
- Outputs land on scratch; rsync the figures/JSONs back to the laptop when done.
- Alternative (if single-node walltime limits are tight): generational array jobs with a driver that submits one array task per candidate and syncs between generations. More moving parts; only if needed.

## What we need from the user / to verify

1. Access: confirm Crescent2 account, ssh alias, and which partition(s) we may use + walltime/core limits (check `sinfo`, `sacctmgr show assoc user=$USER`).
2. Environment: Python with numpy/scipy/matplotlib. Options: (a) cluster `module load python/...` + pip install --user into a venv on shared storage; (b) copy the laptop .venv (risky — glibc/arch mismatch; homebrew python 3.14 almost certainly not on the cluster). Prefer a fresh venv on the cluster.
3. Files to stage: `Analysis/rd_core.py`, `Analysis/RD/rd_train_xor_phi_fast.py`, `Analysis/RD/config.json`. No other dependencies (no Tellurium, no JAX). ~100 KB total.
4. Scratch space for outputs (~10-50 MB per campaign) and a sync-back step.

## Draft SLURM script (template, adjust partition/account)

```bash
#!/bin/bash
#SBATCH --job-name=rd_xor_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96      # set to node size
#SBATCH --mem-per-cpu=1G
#SBATCH --time=04:00:00
#SBATCH --output=rd_train_%j.log

cd $SLURM_SUBMIT_DIR
source ~/venvs/rdtrain/bin/activate   # cluster venv
python rd_train_xor_phi_fast.py train --popsize 4 --maxiter 4 --workers $SLURM_CPUS_PER_TASK
```

(Requires adding a `--workers` flag to the fast script — trivial, one line.)

## Effort estimate

- If access already works: ~1 h setup (venv + rsync + one 10-min test job), then campaigns on demand.
- The laptop stays the dev/sanity environment (protocol validation, smoke tests); the cluster runs full campaigns.

## Recommendation

Worth it if (a) the trimmed local run still projects >8 h per campaign, or (b) we move to campaign 2 (anisotropy training) which doubles the weight space. Otherwise the trimmed local overnight run is adequate for the thesis headline.
