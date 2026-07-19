# 2D Cylinder Flow — OpenFOAM Validation Case

Laminar flow over a circular cylinder at `Re = 100`.

## Physics

- Incompressible, laminar Navier–Stokes.
- Cylinder diameter `D = 0.01 m`.
- Inlet velocity `U = 0.15 m/s`.
- Kinematic viscosity `nu = 1.5e-5 m^2/s` (air).
- Expected Reynolds number: `Re = U*D/nu = 100`.
- Expected Strouhal number: `St = f*D/U ≈ 0.2`.
- Expected shedding frequency: `f ≈ 0.2 * U / D = 3 Hz`.

## Mesh

The `blockMeshDict` creates a half-domain (`y >= 0`) with a C-grid/O-grid around
the cylinder. `mirrorMesh -overwrite` reflects it across `y = 0` to give the
full channel.

Default mesh scaling factor = 4 → ~9500 cells after mirroring.

## Running

```bash
cd Validation/cylinder_array_2d
./Allrun
```

This executes:
1. `blockMesh` — build half-domain mesh.
2. `mirrorMesh -overwrite` — mirror to full domain.
3. `icoFoam` — run transient laminar solver.

## Outputs

- Velocity and pressure time histories at three wake probes:
  `postProcessing/probes/0/U` and `postProcessing/probes/0/p`.
- Lift/drag forces and force coefficients:
  `postProcessing/forceCoeffsIncompressible/0/forceCoeffs.dat`.

## Computing Strouhal number

From `postProcessing/forceCoeffsIncompressible/0/forceCoeffs.dat`, take the
lift coefficient column (4th column) and compute its FFT. The dominant peak
frequency is the vortex-shedding frequency `f`. Then:

```
St = f * D / U
```

## Changing parameters

Edit these files to sweep `Re`:

- `constant/physicalProperties` → `nu`
- `0/U` → `Uinlet`
- `system/blockMeshDict` → `diameter`

To change spacing for multiple cylinders, the `blockMeshDict` must be extended
or replaced with `snappyHexMesh`.

## Notes

- The `dirname: missing operand` warning when sourcing `/opt/openfoam13/etc/bashrc`
  is harmless; it comes from the ParaView setup path.
- Run `bash` first if your default shell is `zsh`.
