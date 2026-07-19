"""
pulse_baseline.py

Run a Gaussian pulse through uniform medium (no grating).
Record time-domain signal at outlet to identify pulse arrival.
"""

import numpy as np
import matplotlib.pyplot as plt
from acoustic_filter_with_thermal import solve_temperature

# =============================================================================
# PARAMETERS
# =============================================================================
NX, NY = 400, 50
DX = 0.002
C0 = 343.0
RHO0 = 1.225
SIM_DURATION = 0.020  # 20 ms

# Source
I_SRC_START = 3
I_SRC_END = 6

# Pulse parameters
PULSE_AMP = 1.0
PULSE_T0 = 0.0005      # pulse center at 0.5 ms
PULSE_TAU = 0.0001     # pulse width 0.1 ms
PULSE_F0 = 6000.0      # carrier frequency 6 kHz

# Outlet measurement point
I_MEASURE = NX - 4
J_MEASURE = NY // 2

# =============================================================================
# BUILD UNIFORM C-FIELD
# =============================================================================
T_uniform = solve_temperature(np.zeros((NX, NY)))
c_field = C0 * np.sqrt(T_uniform / 300.0)

# =============================================================================
# FDTD WITH PULSE SOURCE
# =============================================================================
c_max = float(c_field.max())
dt = 0.9 * DX / (c_max * np.sqrt(2))
n_steps = int(SIM_DURATION / dt)

p = np.zeros((NX, NY))
u = np.zeros((NX + 1, NY))
v = np.zeros((NX, NY + 1))

# Time history
p_history = []
t_history = []

for n in range(n_steps):
    t = n * dt
    
    # Update velocities
    u[1:NX, :] -= (dt / RHO0) * (p[1:NX, :] - p[0:NX - 1, :]) / DX
    v[:, 1:NY] -= (dt / RHO0) * (p[:, 1:NY] - p[:, 0:NY - 1]) / DX
    
    # Absorbing BCs
    u[0, :]  = -p[0, :] / (RHO0 * c_field[0, :])
    u[NX, :] =  p[NX - 1, :] / (RHO0 * c_field[NX - 1, :])
    v[:, 0]  = -p[:, 0] / (RHO0 * c_field[:, 0])
    v[:, NY] =  p[:, NY - 1] / (RHO0 * c_field[:, NY - 1])
    
    # Update pressure
    p -= dt * RHO0 * c_field**2 * (
        (u[1:NX + 1, :] - u[0:NX, :]) / DX +
        (v[:, 1:NY + 1] - v[:, 0:NY]) / DX
    )
    
    # Gaussian pulse source
    envelope = np.exp(-((t - PULSE_T0) / PULSE_TAU)**2)
    p[I_SRC_START:I_SRC_END, :] += PULSE_AMP * envelope * np.sin(2 * np.pi * PULSE_F0 * (t - PULSE_T0))
    
    # Record
    p_history.append(float(p[I_MEASURE, J_MEASURE]))
    t_history.append(t)

p_history = np.array(p_history)
t_history = np.array(t_history)

# =============================================================================
# PLOTS
# =============================================================================
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Time domain
ax = axes[0]
ax.plot(t_history * 1000, p_history, 'b-', linewidth=0.8)
ax.axvline(x=PULSE_T0 * 1000, color='r', linestyle='--', alpha=0.5, label='pulse center (t0)')
ax.axvline(x=(PULSE_T0 + 3*PULSE_TAU) * 1000, color='g', linestyle='--', alpha=0.5, label='t0 + 3*tau')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Pressure p')
ax.set_title('Baseline: Pulse arrival at outlet (no grating)')
ax.grid(True, alpha=0.3)
ax.legend()

# Zoom in on first 5 ms
ax_inset = ax.inset_axes([0.55, 0.55, 0.4, 0.4])
ax_inset.plot(t_history * 1000, p_history, 'b-', linewidth=0.8)
ax_inset.set_xlim(0, 5)
ax_inset.set_title('Zoom: first 5 ms')
ax_inset.grid(True, alpha=0.3)

# FFT
ax = axes[1]
# Window the pulse arrival (0.5-5 ms)
window_start = int(0.0005 / dt)
window_end = int(0.005 / dt)
p_windowed = p_history[window_start:window_end]

# Hann window
hann = np.hanning(len(p_windowed))
p_windowed = p_windowed * hann

# FFT
freqs = np.fft.rfftfreq(len(p_windowed), dt)
spectrum = np.abs(np.fft.rfft(p_windowed))

ax.semilogy(freqs / 1000, spectrum, 'b-', linewidth=1)
ax.axvline(x=PULSE_F0 / 1000, color='r', linestyle='--', alpha=0.5, label=f'f0 = {PULSE_F0/1000:.0f} kHz')
ax.set_xlabel('Frequency (kHz)')
ax.set_ylabel('|P(f)|')
ax.set_title('FFT of windowed pulse (0.5-5 ms)')
ax.set_xlim(0, 10)
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig('pulse_baseline.png', dpi=150)
print("Saved: pulse_baseline.png")
plt.show()

# Print stats
peak_idx = np.argmax(np.abs(p_history))
peak_time = t_history[peak_idx]
print(f"\nPulse peak arrives at: {peak_time*1000:.3f} ms")
print(f"Transit time (estimate): {I_MEASURE*DX/C0*1000:.3f} ms")
