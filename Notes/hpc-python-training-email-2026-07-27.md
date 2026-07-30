# Draft email to HPC support — Python training workloads on Crescent2

Date: 2026-07-27
Status: DRAFT — review before sending. Fill in [bracketed] items.
To: [hpc-support@cranfield.ac.uk / Crescent2 support address]
Subject: Running custom Python (NumPy/SciPy) optimisation workloads on Crescent2 — environment and job setup advice

---

Dear HPC Support Team,

I am an MSc student in the [Centre for Computational Engineering Sciences / department], supervised by [Prof. Weisi Guo / Prof. Takis Tsoutsanis]. I have previously run ANSYS Fluent jobs on Crescent2 using the standard module setup, and I would like some advice on the correct way to run a **custom Python-based workload** for my research project.

## The workload

My project involves training a spatial parameter field for a reaction-diffusion simulation. Concretely:

- A **differential evolution** optimisation loop over ~64 parameters.
- Each candidate evaluation consists of 4 independent single-core simulations (2D finite-difference solver, NumPy/SciPy, ~1–4 minutes each, ~100 MB RAM).
- One generation = 256 candidates = 1024 single-core sims; a full campaign is ~5 generations.
- Simulations are independent within a generation (embarrassingly parallel); the only synchronisation point is the optimiser between generations.
- No GPU, no MPI, no licensed software. Dependencies: Python 3 with `numpy`, `scipy`, `matplotlib` only.

On my workstation I currently parallelise a generation over a process pool (Python `multiprocessing`) on 8–16 cores.

## My questions

1. **Python environment:** What is the recommended way to set up Python with NumPy/SciPy on Crescent2? Is there a supported `python`/`scipy` module, or should I create a virtual environment (or conda env) in my home directory? Are `pip install --user` / `venv` installs permitted?
2. **Job shape:** For this workload, is it preferred to submit (a) a single-node job using all cores of the node with my internal process pool, or (b) a SLURM array job with one array task per candidate? Are there walltime or core-count limits I should design around?
3. **Storage:** Where should simulation outputs (tens of MB of JSON/PNG per campaign) be written — home or scratch — and what are the quotas?
4. **Access and accounting:** My account is [username]. Is there anything I need to request (partition access, accounting group) before submitting?

A short example SLURM script for a Python workload of this kind would be very helpful if one is available.

Thank you for your time — happy to provide more detail on the workload if useful.

Best regards,
Srivijayesh Venugopal
MSc [Computational Fluid Dynamics], Cranfield University
[student ID] | [email]

---

## Notes for us (delete before sending)

- Attach or link `Notes/hpc-readiness-2026-07-27.md` internally for our own reference; do not send.
- If they recommend array jobs, the DE driver needs the generational-sync pattern (submit generation as array, collect, submit next) — slightly more work; single-node-with-pool stays the preference.
- If a module Python is older than 3.10, check `rd_core.py` compatibility before relying on it (it uses only standard NumPy/SciPy, should be fine back to ~3.9).
