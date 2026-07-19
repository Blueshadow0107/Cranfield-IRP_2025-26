"""
small_spacing_sweep_fixed.py

Fixed: dt computed from c_field.max() (485 m/s) not c_ref.max() (343 m/s).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from acoustic_filter_with_thermal import solve_temperature

NX, NY = 400, 50
DX = 0.002
C0 = 343.0
RHO0 = 1.225
SIM_DURATION = 0.020
I_SRC_START = 3
I_SRC_END = 6
PULSE_AMP = 1.0
PULSE_T0 = 0.0005
PULSE_TAU = 0.00006
PULSE_F0 = 8000.0
I_MEASURE = NX - 4
J_MEASURE = NY // 2
WIN_START = 0.0020
WIN_END = 0.0035
T_HOT = 600.0
C_HOT = C0 * np.sqrt(T_HOT / 300.0)
C_COLD = C0
STRIP_WIDTH = 0.010
N_PERIODS = 20
START_X = 0.060
PERIODS = np.array([0.018, 0.020, 0.022, 0.024, 0.026, 0.028])

print("Running reference (no grating)...")
T_uniform = solve_temperature(np.zeros((NX, NY)))
c_ref = C0 * np.sqrt(T_uniform / 300.0)

# FIX: compute dt from actual max c (hot strips)
c_max = float(C_HOT)
dt = 0.9 * DX / (c_max * np.sqrt(2))
n_steps = int(SIM_DURATION / dt)
print(f"dt = {dt:.2e}, n_steps = {n_steps}")

p = np.zeros((NX, NY))
u = np.zeros((NX + 1, NY))
v = np.zeros((NX, NY + 1))
p_ref = []
for n in range(n_steps):
    t = n * dt
    u[1:NX, :] -= (dt / RHO0) * (p[1:NX, :] - p[0:NX - 1, :]) / DX
    v[:, 1:NY] -= (dt / RHO0) * (p[:, 1:NY] - p[:, 0:NY - 1]) / DX
    u[0, :]  = -p[0, :] / (RHO0 * c_ref[0, :])
    u[NX, :] =  p[NX - 1, :] / (RHO0 * c_ref[NX - 1, :])
    v[:, 0]  = -p[:, 0] / (RHO0 * c_ref[:, 0])
    v[:, NY] =  p[:, NY - 1] / (RHO0 * c_ref[:, NY - 1])
    p -= dt * RHO0 * c_ref**2 * ((u[1:NX + 1, :] - u[0:NX, :]) / DX + (v[:, 1:NY + 1] - v[:, 0:NY]) / DX)
    envelope = np.exp(-((t - PULSE_T0) / PULSE_TAU)**2)
    p[I_SRC_START:I_SRC_END, :] += PULSE_AMP * envelope * np.sin(2 * np.pi * PULSE_F0 * (t - PULSE_T0))
    p_ref.append(float(p[I_MEASURE, J_MEASURE]))

p_ref = np.array(p_ref)
win_s = int(WIN_START / dt)
win_e = int(WIN_END / dt)
p_ref_w = p_ref[win_s:win_e] * np.hanning(win_e - win_s)
freqs = np.fft.rfftfreq(len(p_ref_w), dt)
spec_ref = np.abs(np.fft.rfft(p_ref_w))

colors = plt.cm.viridis(np.linspace(0, 1, len(PERIODS)))
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

for idx, period in enumerate(PERIODS):
    print(f"Running period = {period*1000:.0f} mm...")
    c_field = np.ones((NX, NY)) * C_COLD
    period_cells = int(period / DX)
    width_cells = int(STRIP_WIDTH / DX)
    start_i = int(START_X / DX)
    for n in range(N_PERIODS):
        i_s = start_i + n * period_cells
        i_e = i_s + width_cells
        if i_e < NX:
            c_field[i_s:i_e, :] = C_HOT
    p = np.zeros((NX, NY))
    u = np.zeros((NX + 1, NY))
    v = np.zeros((NX, NY + 1))
    p_grating = []
    for n in range(n_steps):
        t = n * dt
        u[1:NX, :] -= (dt / RHO0) * (p[1:NX, :] - p[0:NX - 1, :]) / DX
        v[:, 1:NY] -= (dt / RHO0) * (p[:, 1:NY] - p[:, 0:NY - 1]) / DX
        u[0, :]  = -p[0, :] / (RHO0 * c_field[0, :])
        u[NX, :] =  p[NX - 1, :] / (RHO0 * c_field[NX - 1, :])
        v[:, 0]  = -p[:, 0] / (RHO0 * c_field[:, 0])
        v[:, NY] =  p[:, NY - 1] / (RHO0 * c_field[:, NY - 1])
        p -= dt * RHO0 * c_field**2 * ((u[1:NX + 1, :] - u[0:NX, :]) / DX + (v[:, 1:NY + 1] - v[:, 0:NY]) / DX)
        envelope = np.exp(-((t - PULSE_T0) / PULSE_TAU)**2)
        p[I_SRC_START:I_SRC_END, :] += PULSE_AMP * envelope * np.sin(2 * np.pi * PULSE_F0 * (t - PULSE_T0))
        p_grating.append(float(p[I_MEASURE, J_MEASURE]))
    p_grating = np.array(p_grating)
    p_grating_w = p_grating[win_s:win_e] * np.hanning(win_e - win_s)
    spec_grating = np.abs(np.fft.rfft(p_grating_w))
    transmission = spec_grating / (spec_ref + 1e-30)
    axes[0].semilogy(freqs / 1000, transmission, color=colors[idx], linewidth=1.5, label=f'd = {period*1000:.0f} mm')
    f = STRIP_WIDTH / period  # fill fraction (varies with period!)
    n_eff = f * (C_COLD / C_HOT) + (1.0 - f) * 1.0
    f_bragg = C_COLD / (2 * n_eff * period)
    axes[0].axvline(x=f_bragg / 1000, color=colors[idx], linestyle='--', alpha=0.3)

axes[0].set_xlabel('Frequency (kHz)')
axes[0].set_ylabel('Transmission')
axes[0].set_title('Transmission vs Frequency for Different Grating Periods')
axes[0].set_xlim(0, 12)
axes[0].grid(True, alpha=0.3)
axes[0].legend(loc='upper right')

axes[1].axis('off')
table_data = []
for period in PERIODS:
    f = STRIP_WIDTH / period
    n_eff = f * (C_COLD / C_HOT) + (1.0 - f) * 1.0
    f_bragg = C_COLD / (2 * n_eff * period)
    table_data.append([f'{period*1000:.0f} mm', f'{f_bragg/1000:.1f} kHz'])
table = axes[1].table(cellText=table_data, colLabels=['Period', 'Bragg Frequency (approx)'], cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2)
axes[1].set_title('Bragg Condition: f = c / (2 * n_eff * d)', pad=20)
plt.tight_layout()
plt.savefig('small_spacing_sweep_fixed.png', dpi=150)
print("Saved: small_spacing_sweep_fixed.png")
