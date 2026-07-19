#FDTD trial

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.signal import find_peaks

# Physical parameters

#Grid 
Nx = 400
Ny = 50 
Dx = 0.002 # m (2 mm)
Dy = Dx

Lx = Nx*Dx
Ly = Ny*Dy

#Air at 300 K
Rho0 = 1.225 # density [kg/m^3]
C0 = 343.0 # speed of sound [m/s]
T0 = 300.0 # temperature [K]
D_THERMAL = 2e-5 # [m^2/s] #Thermal diffusivity of air


# Build the sound speed field
def c_field_uniform():
    return np.full((Nx, Ny), C0)

def Temp_solver(sigma, dx=Dx, D=D_THERMAL, T0=T0):
    nx, ny = sigma.shape
    N = nx*ny

    #build Laplacian
    rows, cols, data = [], [], []
    for i in range(nx):
        for j in range(ny):
            k = i*ny+j #flatten 2D index (i,j) -> 1D index k 
            if i == 0 or i == nx-1 or j == 0 or j == ny-1: #boundary cell
                rows.append(k)
                cols.append(k)
                data.append(1.0)
            else: #interior cell
                rows.append(k); cols.append(k); data.append(-4.0) #centre
                rows.append(k); cols.append((i+1)*ny+j); data.append(1.0) #neighbour east
                rows.append(k); cols.append((i-1)*ny+j); data.append(1.0) #neighbour west
                rows.append(k); cols.append(i*ny+(j+1)); data.append(1.0) #neighbour north
                rows.append(k); cols.append(i*ny+(j-1)); data.append(1.0) #neighbour south

    A = csr_matrix((data, (rows, cols)), shape=(N, N))


    b = np.zeros(N)
    for i in range(nx):
        for j in range(ny):
            k = i*ny+j
            if i == 0 or i == nx-1 or j == 0 or j == ny-1:
                b[k] = T0
            else:
                b[k] = -sigma[i,j]*dx**2/D

    T_flat = spsolve(A, b)
    return T_flat.reshape((nx, ny))

    def heat_source(sigma_amps, pos, wid):
        sigma = np.zeros((Nx, Ny))
        half_w_cells = int(wid / (2*Dx))
        for amp, pos in zip(sigma_amps, pos):
            i_c = int(pos / Dx)
            i_s = max(0, i_c - half_w_cells)
            i_e = min(Nx, i_c + half_w_cells)
            sigma[i_s:i_e, :] = amp
        return sigma

    def c_field_from_thermal(sigma_amps, pos, wid):
        sigma = heat_source(sigma_amps, pos, wid)
        T = Temp_solver(sigma)
        return C0 * np.sqrt(T / T0)

#Initialisation

c_field = c_field_uniform()
c2 = c_field ** 2

p = np.zeros((Nx, Ny))
u = np.zeros((Nx + 1, Ny))
v = np.zeros((Nx, Ny + 1))
c_max = float(c_field.max())
dt = 0.9 * Dx / (c_max * np.sqrt(2))
print(f"dt = {dt*1e6:.3f} µs, Courant = {c_max*dt/Dx:.4f}")

BC_LEFT = 'absorbing'
BC_RIGHT = 'absorbing'
BC_BOTTOM = 'hard_wall'
BC_TOP = 'hard_wall'


def boundary_conditions():
    global u, v

    # LEFT face (x = 0)
    if BC_LEFT == 'absorbing':
        u[0, :] = -p[0, :] / (Rho0 * c_field[0, :]) #Engquist-Majda 
    elif BC_LEFT == 'hard_wall':
        u[0, :] = 0.0 #zero normal velocity
    elif BC_LEFT == 'pressure_release':
        u[0, :] -= (dt / Rho0) * 2.0 * p[0, :] / Dx #P = 0 at boundary 

    # RIGHT face (x = L)
    if BC_RIGHT == 'absorbing':
        u[Nx, :] = p[Nx - 1, :] / (Rho0 * c_field[Nx - 1, :]) #Engquist-Majda 
    elif BC_RIGHT == 'hard_wall':
        u[Nx, :] = 0.0 #zero normal velocity
    elif BC_RIGHT == 'pressure_release':
        u[Nx, :] += (dt / Rho0) * 2.0 * p[Nx - 1, :] / Dx #P = 0 at boundary 

    # BOTTOM face (y = 0)
    if BC_BOTTOM == 'absorbing':
        v[:, 0] = -p[:, 0] / (Rho0 * c_field[:, 0]) #Engquist-Majda 
    elif BC_BOTTOM == 'hard_wall':
        v[:, 0] = 0.0 #zero normal velocity
    elif BC_BOTTOM == 'pressure_release':
        v[:, 0] -= (dt / Rho0) * 2.0 * p[:, 0] / Dy #P = 0 at boundary 

    # TOP face (y = L)
    if BC_TOP == 'absorbing':
        v[:, Ny] = p[:, Ny - 1] / (Rho0 * c_field[:, Ny - 1]) #Engquist-Majda 
    elif BC_TOP == 'hard_wall':
        v[:, Ny] = 0.0 #zero normal velocity
    elif BC_TOP == 'pressure_release':
        v[:, Ny] += (dt / Rho0) * 2.0 * p[:, Ny - 1] / Dy #P = 0 at boundary 


# Source injection

SOURCE_TYPE = 'continuous_sine'   # options: 'continuous_sine', 'gaussian_pulse', 'broadband_pulse'
SRC_I0, SRC_I1 = 3, 6             # x-cells where source is injected

# Continuous sine parameters
SRC_FREQ = 5000.0                 # Hz
SRC_AMP = 1.0

# Gaussian pulse parameters
PULSE_T0 = 0.0005                 # pulse centre time [s]
PULSE_TAU = 0.0001                # pulse width [s]
PULSE_F0 = 6000.0                 # carrier frequency [Hz]

def source_amplitude(t):
    if SOURCE_TYPE == 'continuous_sine':
        return SRC_AMP * np.sin(2.0 * np.pi * SRC_FREQ * t)
    elif SOURCE_TYPE == ('gaussian_pulse'):
        envelope = np.exp(-((t - PULSE_T0) / PULSE_TAU) ** 2)
        return SRC_AMP * envelope * np.sin(2.0 * np.pi * PULSE_F0 * (t - PULSE_T0))
    elif SOURCE_TYPE == ('broadband_pulse'):
        envelope = np.exp(-((t - PULSE_T0) / PULSE_TAU*2) ** 2)
        return SRC_AMP * envelope * np.sin(2.0 * np.pi * PULSE_F0/3 * (t - PULSE_T0))
    else:
        return 0.0


# Time-stepping

def step(t):
    global p, u, v

    # Update interior velocities
    u[1:Nx, :] -= (dt / Rho0) * (p[1:Nx, :] - p[0:Nx-1, :]) / Dx
    v[:, 1:Ny] -= (dt / Rho0) * (p[:, 1:Ny] - p[:, 0:Ny-1]) / Dy

    # Apply boundary conditions
    boundary_conditions()

    # Update pressure
    du_dx = (u[1:Nx+1, :] - u[0:Nx, :]) / Dx
    dv_dy = (v[:, 1:Ny+1] - v[:, 0:Ny]) / Dy
    p -= dt * Rho0 * c2 * (du_dx + dv_dy)

    # Source injection
    src = source_amplitude(t)
    p[SRC_I0:SRC_I1, :] += src

# Runnnnn

duration = 0.01 #seconds
n_steps = int(duration / dt)

probloc = 'outlet'

def probe_placement(probloc):
    if probloc == 'outlet':
        return Nx - 4, Ny // 2
    elif probloc == 'centred':
        return Nx // 2, Ny // 2
    elif probloc == 'inlet':
        return 4, Ny // 2
    else:
        raise ValueError(f"Invalid probe location: {probloc}")

probe_i, probe_j = probe_placement(probloc)
probe_history = []

print(f"Running {n_steps} steps ({duration*1000:.1f} ms)...")

for n in range(n_steps):
    t = n * dt
    step(t) 

    probe_history.append(float(p[probe_i, probe_j]))

    if n % 2000 == 0:
        print(f"  Step {n}/{n_steps}, t = {t*1000:.2f} ms")

probe_history = np.array(probe_history)
t_history = np.arange(n_steps) * dt

print("Done.")


# Visualisation

fig, axes = plt.subplots(2, 1, figsize=(10, 6))

#time
ax = axes[0]
ax.plot(t_history * 1000, probe_history, 'b-', linewidth=0.5)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Pressure p')
ax.set_title(f'Probe at cell ({probe_i},{probe_j})')
ax.grid(True, alpha=0.3)

#FFT
ax = axes[1]
window = np.hanning(len(probe_history))
freqs = np.fft.rfftfreq(len(probe_history), dt)
spectrum = np.abs(np.fft.rfft(probe_history * window))
ax.semilogy(freqs / 1000, spectrum, 'b-', linewidth=0.5)
ax.set_xlabel('Frequency (kHz)')
ax.set_ylabel('|P(f)|')
ax.set_title('FFT of probe signal')
ax.set_xlim(0, 12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fdtd_output.png', dpi=150)
print("Saved: fdtd_output.png")
plt.show()

#Verification 

Tests = ['pulse_arrival', 'hard_wall', 'pressure_release', 'eigenfrequencies', 'energy_conservation', 'phase_velocity']

