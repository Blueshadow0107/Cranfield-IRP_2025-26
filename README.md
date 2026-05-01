# Cranfield IRP 2025-26

Individual Research Project for the MSc programme at Cranfield University.

## Structure

| Directory | Purpose |
|-----------|---------|
| `Report/` | LaTeX thesis |
| `Analysis/` | MATLAB/Python analysis scripts |
| `Validation/` | CFD cases, mesh studies, solver comparisons |
| `Design/` | CAD geometries, OpenVSP models |
| `Notes/` | Research notes, literature reviews |
| `Presentations/` | Progress presentations |
| `Scripts/` | Automation and HPC submission scripts |

## Quick Start

```bash
# Compile thesis
cd Report && latexmk -pdf main

# Submit HPC job
cd Validation/<case> && sbatch ../Scripts/hpc_submit_template.sh
```

## Git

```bash
git clone https://github.com/Blueshadow0107/Cranfield-IRP_2025-26.git
cd Cranfield-IRP_2025-26
```
