"""rd_demux_baseline_anim.py -- baseline demux run (uniform phi0) with VTK
frame export for ParaView visualisation.

Runs the three input patterns (10), (01), (11) at uniform phi = 0.010 on
the locked V5 geometry (imported from rd_demux_train), records the probe
traces, and writes the activator field u(x, y) as a .vti time series plus
a .pvd collection per pattern (open the .pvd in ParaView and press play).

Frames: every STRIDE steps (0.5 t.u. at DT=0.05), float32, ~23 MB per
pattern.  Output: Analysis/figures/demux_baseline_anim/
"""

import json
import os

import numpy as np

import rd_demux_train as dm

from xml.etree.ElementTree import Element, SubElement, ElementTree
import vtk
from vtk.util.numpy_support import numpy_to_vtk

OUT_DIR = os.path.join(dm.FIG, 'demux_baseline_anim')
os.makedirs(OUT_DIR, exist_ok=True)

STRIDE = 10                      # steps between frames (0.5 t.u.)


def write_vti(path, u, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    """Write 2D numpy array u[x, y] to a .vti file using VTK."""
    nx, ny = u.shape

    img = vtk.vtkImageData()
    img.SetDimensions(nx, ny, 1)
    img.SetSpacing(*spacing)
    img.SetOrigin(*origin)

    arr = np.ascontiguousarray(u.T.ravel(order="C"), dtype=np.float32)
    vtk_arr = numpy_to_vtk(arr, deep=True, array_type=vtk.VTK_FLOAT)
    vtk_arr.SetName("u")
    img.GetPointData().SetScalars(vtk_arr)

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(path)
    writer.SetInputData(img)
    writer.SetDataModeToBinary()
    writer.Write()

def write_pvd(path, frames):
    """Write a ParaView collection file.
    frames: list of (time, relative_filename)
    """
    root = Element("VTKFile", type="Collection", version="0.1", byte_order="LittleEndian")
    coll = SubElement(root, "Collection")

    for t, fname in frames:
        SubElement(coll, "DataSet", timestep=str(float(t)), group="", part="0", file=fname)

    ElementTree(root).write(path, encoding="UTF-8", xml_declaration=True)

def run_pattern_anim(tag):
    fa, fb = dm.FIRE[tag]
    phi = np.full((dm.NX, dm.NY), dm.PHI0)
    rd = dm.make_rd(phi)
    phi_dark = phi.copy()
    if fa:
        phi_dark[dm.spot_mask(dm.A_POS)] = dm.DARK
    if fb:
        phi_dark[dm.spot_mask(dm.B_POS)] = dm.DARK

    case_dir = os.path.join(OUT_DIR, f'pattern_{tag}')
    os.makedirs(case_dir, exist_ok=True)
    frames = []
    probe_series = {name: [] for name in dm.PROBES}

    def snapshot(step):
        if step % STRIDE == 0:
            fname = f'frame_{step:05d}.vti'
            write_vti(os.path.join(case_dir, fname), rd.u)
            frames.append((step * dm.DT, fname))

    # flash phase (phi_dark for T_FLASH)
    rd.set_phi(phi_dark)
    for s in range(int(dm.T_FLASH / dm.DT)):
        rd.run(1)
        for name, pos in dm.PROBES.items():
            probe_series[name].append(float(rd.u[pos]))
        snapshot(s)
    # restore + free evolution
    rd.set_phi(phi)
    n0 = int(dm.T_FLASH / dm.DT)
    for s in range(int(dm.TU_AFTER / dm.DT)):
        rd.run(1)
        for name, pos in dm.PROBES.items():
            probe_series[name].append(float(rd.u[pos]))
        snapshot(n0 + s)

    write_pvd(os.path.join(OUT_DIR, f'pattern_{tag}.pvd'),
              [(t, f'pattern_{tag}/{f}') for t, f in frames])
    return {k: [float(v) for v in np.array(vs)[::STRIDE]]
            for k, vs in probe_series.items()}


def main():
    traces = {}
    for tag in dm.PATTERNS:
        print(f'[baseline] pattern ({tag}) ...', flush=True)
        traces[tag] = run_pattern_anim(tag)
        print(f'[baseline] pattern ({tag}) done', flush=True)
    with open(os.path.join(OUT_DIR, 'baseline_traces.json'), 'w') as fh:
        json.dump({'dt': dm.DT, 'stride': STRIDE, 'phi0': dm.PHI0,
                   'patterns': traces}, fh)
    print(f'[done] {OUT_DIR} (open pattern_*.pvd in ParaView)', flush=True)


if __name__ == '__main__':
    main()
