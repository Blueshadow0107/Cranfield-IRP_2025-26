import numpy as np 
import matplotlib.pyplot as plt   
from scipy.signal import find_peaks
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from pathlib import Path 

Nx = 400 
Ny = 50 
Dx = 0.002 # m
Dy = 0.002 # m
C0 = 343.0 # m/s 
Rho0 = 122.5 # kg/m^3 
T0 = 300 # K
DThermal = 2e-5 # m^2/s  

#sound nd time 
c_field = None
c2 = None
cmax = None
dt = None

#Fields 
p = None 
u = None
v = None 

#Default Boundary Condition
BcLft = 'OpenWorld'
BcRgt = 'OpenWorld'
BcTop = 'Wall'
BcBtm = 'Wall'

SrcType  = 'sine'
PulsType = 'Gaussian'
SrcI0 = 3
SrcI1 = 6
SrcAmp = 1.0
SrcFrq = 5000.0 #Hz 
PulsT0 = 0.0005 #s
PulsTau = 0.0001 #Pulse Activation time

def HeatSrcBldr(SigmAmp, positions, width):
    sigma = np.zeros((Nx, Ny))
    HalfCellWidth = int(width / (2*Dx))
    for amp, pos in zip(SigmAmp, positions):
        Ic = int(pos / Dx)
        Is = max(0, Ic - HalfCellWidth)
        Ie = min(Nx, Ic + HalfCellWidth)
        sigma[Is:Ie, :] = amp 
    return sigma

def TempSlvr(sigma):
    nx, ny = sigma.shape 
    N = nx*ny
    b = np.zeros
    rows, cols, data = [], [], []
    for i in range(nx):
        for j in range(ny):
            k = i*ny + j 
            if i == 0 or i == nx or j == 0 or j == ny:
                #Boundary Cell
                rows.append(k) 
                cols.append(k)
                data.append(1.0)
                b[k] = T0
            else: 
                #5 Point Stencil
                rows.append(k) 
                cols.append(k)
                data.append(-4.0)

                rows.append(k)
                cols.append(k+ny)
                data.append(1.0)

                rows.append(k)
                cols.append(k-ny)
                data.append(1.0)

                rows.append(k)
                cols.append(k+1)
                data.append(1.0)

                rows.append(k)
                cols.append(k-1)
                data.append(1.0)

                b[k] = -sigma[i,j] * Dx ** 2 / DThermal

    A = csr_matrix((data, (rows,cols)), shape=(N,N))
    TFlat = spsolve(A,b)
    return TFlat.reshape((nx,ny))


def CBldr(SigmAmps, positions, width):
    sigma = HeatSrcBldr(SigmAmps, positions, width)
    T = TempSlvr(sigma)

    return C0 * np.sqrt(T/T0)

def CSet(NewCField):
    global CField, C2, CMax, dt
    CField = NewCField.astype(float)
    C2 = CField ** 2
    CMax = float(CField.max())
    dt = 0.9 * Dx / (CMax * np.sqrt(2))
    print(f" CMax = {CMax:.1f} m/s, dt = {dt*1e6:.3f} microseconds, CFL = {CMax*dt/Dx:.4f}")

def reset():
    global u, v, p
    p = np.zeros((Nx, Ny))
    u = np.zeros((Nx+1, Ny))
    v = np.zeros((Nx, Ny+1))

def ApplyBC():
    global u, v

    if BcLft == 'OpenWorld':
        u[0, :] = -p[0, :] / (Rho0*CField[0,:])
    elif BcLft == 'Wall':
        u[0, :] = 0.0
    elif BcLft == 'PRelease':
        u[0, :] -= (dt / Rho0)*2.0*p[0, :] / Dx

    if BcRgt == 'OpenWorld':
        u[Nx, :] = -p[0, :] / (Rho0*CField[0,:])
    elif BcRgt == 'Wall':
        u[Nx, :] = 0.0
    elif BcRgt == 'PRelease':
        u[Nx, :] -= (dt / Rho0)*2.0*p[0, :] / Dx

    if BcTop == 'OpenWorld':
        v[:, Ny] = -p[0, :] / (Rho0*CField[0,:])
    elif BcTop == 'Wall':
        v[:, Ny] = 0.0
    elif BcTop == 'PRelease':
        v[:, Ny] -= (dt / Rho0)*2.0*p[0, :] / Dx

    if BcBtm == 'OpenWorld':
        v[:, 0] = -p[0, :] / (Rho0*CField[0,:])
    elif BcBtm == 'Wall':
        v[:, 0] = 0.0
    elif BcBtm == 'PRelease':
        v[:, 0] -= (dt / Rho0)*2.0*p[0, :] / Dx

def SrcAmpInj(t):
    if SrcType == 'none':
        return 0.0 
    elif SrcType == 'sine':
        return SrcAmp * np.sin(2.0 * np.pi * SrcFrq * t)
    elif SrcType == 'Pulse': 
        if PulsType == 'Gaussian':
            PulsTau = 0.0001
        elif PulsType == 'Broadband': 
            PulsTau = 0.0005 
        env = np.exp(-((t - PulsT0) / PulsTau) ** 2)
        return SrcAmp * env * np.sin(2.0*np.pi*SrcFrq * (t - PulsT0))
    return 0.0 

def TimeStp(t):
    global p 
    u[1:Nx, :] += (dt/Rho0)*(p[1:Nx, :]-p[0:Nx-1, :])/Dx 
    v[:, 1:Ny] += (dt/Rho0)*(p[:, 1:Ny]-p[:, 0:Ny-1])/Dy #Interior Update
    ApplyBC() #Boundaryyy conditions
    DuDx = (u[1:Nx+1, :] - u[0:Nx, :]) / Dx #Pressure Update
    DvDy = (v[:, 1:Ny+1] - v[:, 0:Ny]) / Dy
    p -= dt*Rho0 * C2 * (DuDx + DvDy)

    p[SrcI0:SrcI1, :] += SrcAmpInj(t) #Source Injection

def Run(Durn, ProbeCoords=None):
    NSteps = int(Durn / dt)
    probe = [] #Data log
    for n in range(NSteps):
        TimeStp(n*dt) 
        if ProbeCoords is not None:
            probe.append(float(p[ProbeCoords[0], ProbeCoords[1]]))
    return np.array(probe)

def EnrgComp(): 
    PotE = np.sum(p ** 2 / (2.0 * Rho0 * C2)) * (Dx ** 2) 
    UAvg = 0.5 * (u[0:Nx, :] ** 2 + u[1:Nx+1, :] ** 2)
    VAvg = 0.5 * (v[:, 0:Ny] ** 2 + v[:, 1:Ny+1] ** 2)
    KinE = 0.5 * Rho0 * np.sum(UAvg + VAvg) * (Dx ** 2)
    return PotE, KinE, PotE + KinE

def FFTClc(probe):
    window = np.hanning(len(probe))
    freqs = np.fft.rfftfreq(len(probe), dt)
    spectrum = np.abs(np.fft.rfft(probe * window))
    return freqs, spectrum

def FFTPeaks(probe, HgtRt = 0.005, Dst = 5): 
    freqs, spectrum = FFTClc(probe)
    prom = np.max(spectrum) * HgtRt
    peaks, _ = find_peaks(spectrum, prominence=prom, Dst=Dst)
    return freqs[peaks], spectrum[peaks]

FigDir = Path(__file__).parent / 'figures'
FigDir.mkdir(exist_ok=True)

def PrbPlt(t, probe, title, filename):
    fig, axes = plt.subplots(2,1, figsize=(10,6))

    axes[0].plot(t * 1000, probe, 'b-', lw=0.5)
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Pressure p')
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)

    freqs, spectrum = FFTClc(probe)
    axes[1].semilogy(freqs / 1000, spectrum, 'b-', lw=0.5)
    axes[1].set_xlabel('Frequency (kHz)')
    axes[1].set_ylabel('|P(f)|')
    axes[1].set_title('FFT')
    axes[1].set_xlim(0, 12)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
