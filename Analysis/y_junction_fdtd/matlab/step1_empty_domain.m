%% Y-Junction Acoustic Logic Unit — 2D FDTD Prototype (Step 1)
% Empty-domain wave propagation with boundary mask system.
%
% Implements the 2D scalar acoustic wave equation:
%     ∂²p/∂t² = c² (∂²p/∂x² + ∂²p/∂y²)
%
% using second-order central differences in space and time (leapfrog).
% Boundary conditions are handled via a bc_mask array for flexibility.
%
% Boundary mask values:
%     0 = interior         (full 5-point stencil)
%     1 = Neumann (rigid)  (∂p/∂n = 0, ghost cell = boundary cell)
%
% Physical parameters (water-like, scaled for fast prototyping):
%     - Domain: 100 x 100 grid points
%     - Grid spacing dx = 100 µm
%     - Wave speed c = 1500 m/s (water)
%     - Time step dt set by CFL = 0.5
%
% Author: MSc IRP 2025-26
% Date: 2026-05-22

clear; clc; close all;

%% Physical and numerical parameters
NX = 100;           % Grid size x
NY = 100;           % Grid size y
DX = 1e-4;          % Grid spacing [m] = 100 µm
C = 1500.0;         % Wave speed [m/s] (water)
CFL = 0.5;          % CFL safety factor (< 1/sqrt(2) for stability)
DT = CFL * DX / (C * sqrt(2));  % Time step [s]

% Stability check
dt_max = DX / (C * sqrt(2));
assert(DT <= dt_max, 'CFL violation: DT=%.3e > dt_max=%.3e', DT, dt_max);
fprintf('dx = %.3e m,  dt = %.3e s,  c = %.0f m/s\n', DX, DT, C);
fprintf('CFL = %.4f  (stability limit = %.4f)\n', C*DT/DX, 1/sqrt(2));

%% Boundary condition mask
% bc_mask(i,j) encodes the boundary condition type at each cell.
% For now: all domain edges are Neumann (rigid wall), everything else interior.

bc_mask = zeros(NX, NY, 'int32');   % 0 = interior by default
bc_mask(1, :) = 1;                  % left edge   → Neumann
bc_mask(end, :) = 1;                % right edge  → Neumann
bc_mask(:, 1) = 1;                  % bottom edge → Neumann
bc_mask(:, end) = 1;                % top edge    → Neumann

fprintf('Boundary cells: %d  (Neumann)\n', sum(bc_mask(:) == 1));
fprintf('Interior cells: %d\n', sum(bc_mask(:) == 0));

%% Laplacian with boundary-aware stencil
% (Local function at bottom of script)

%% Source parameters (Gaussian pulse)
SRC_X = round(NX / 2);   % Source at domain centre (1-based)
SRC_Y = round(NY / 2);
PULSE_WIDTH = 20 * DT;   % Gaussian width [s]
PULSE_DELAY = 50 * DT;   % Time delay before peak [s]
SRC_AMP = 1.0;           % Peak amplitude [Pa]

%% FDTD update coefficient
% p^{n+1} = 2*p^n - p^{n-1} + (c*dt/dx)^2 * Laplacian(p^n)
coef = (C * DT / DX)^2;

%% Field arrays
p_prev = zeros(NX, NY);   % p^{n-1}
p_curr = zeros(NX, NY);   % p^n
p_next = zeros(NX, NY);   % p^{n+1}

%% Time stepping
NT = 400;                   % Total time steps
snapshot_interval = 50;     % Save a figure every N steps
snapshots = cell(1, ceil(NT/snapshot_interval) + 1);
times = zeros(1, ceil(NT/snapshot_interval) + 1);
snap_idx = 0;

fprintf('\nRunning %d time steps...\n', NT);
for n = 0:NT-1
    t = n * DT;
    
    % Inject source (hard source: overwrite the cell)
    p_curr(SRC_X, SRC_Y) = p_curr(SRC_X, SRC_Y) + source_pulse(t, SRC_AMP, PULSE_DELAY, PULSE_WIDTH);
    
    % Compute Laplacian with boundary-aware stencil
    laplacian = compute_laplacian(p_curr, bc_mask);
    
    % Time update (leapfrog)
    p_next = 2 * p_curr - p_prev + coef * laplacian;
    
    % Swap arrays for next step (Python: p_prev, p_curr, p_next = p_curr, p_next, p_prev)
    temp = p_prev;
    p_prev = p_curr;
    p_curr = p_next;
    p_next = temp;  % Reuse old p_prev array
    
    % Save snapshot
    if mod(n, snapshot_interval) == 0
        snap_idx = snap_idx + 1;
        snapshots{snap_idx} = p_curr;
        times(snap_idx) = t;
    end
    
    % Progress
    if mod(n, 100) == 0
        fprintf('  Step %4d/%d  t = %.2f µs  max |p| = %.4f\n', n, NT, t*1e6, max(abs(p_curr(:))));
    end
end
fprintf('Done.\n\n');

snapshots = snapshots(1:snap_idx);
times = times(1:snap_idx);

%% Visualisation
scriptPath = mfilename('fullpath');
if isempty(scriptPath)
    figDir = fullfile(pwd, 'figures');
else
    [scriptDir, ~, ~] = fileparts(scriptPath);
    figDir = fullfile(scriptDir, '..', 'figures');  % Parent of matlab/ folder
end
if ~exist(figDir, 'dir')
    mkdir(figDir);
end

n_snaps = length(snapshots);
ncols = 3;
nrows = ceil(n_snaps / ncols);
figure('Position', [100 100 300*ncols 250*nrows], 'Color', 'w');

vmax = 0;
for k = 1:n_snaps
    vmax = max(vmax, max(abs(snapshots{k}(:))));
end

x = linspace(0, NX*DX*1e3, NX);
y = linspace(0, NY*DX*1e3, NY);

for i = 1:n_snaps
    subplot(nrows, ncols, i);
    imagesc(x, y, snapshots{i}.');
    axis image;
    set(gca, 'YDir', 'normal');
    colormap(gca, rdBu_r(256));
    caxis([-vmax vmax]);
    colorbar;
    title(sprintf('t = %.1f µs', times(i)*1e6));
    xlabel('x [mm]');
    ylabel('y [mm]');
end
sgtitle('Step 1: Empty-domain wave propagation (Neumann BCs)', 'FontSize', 14);
outPath = fullfile(figDir, 'step1_empty_domain.png');
print(gcf, outPath, '-dpng', '-r150');
fprintf('Saved figure: %s\n', outPath);

%% Debug: Verify 1/sqrt(r) decay in 2D
final = snapshots{end};
r = 1:min(NX, NY)/2 - 1;
p_radial = [];
for ri = r
    ix = SRC_X + ri;
    iy = SRC_Y + ri;
    if ix <= NX && iy <= NY
        p_radial(end+1) = abs(final(ix, iy));
    else
        break;
    end
end
r = r(1:length(p_radial));

figure('Position', [100 100 700 400], 'Color', 'w');
plot(r * DX * 1e3, p_radial, 'b-', 'LineWidth', 1.5, 'DisplayName', '|p(r)|');
hold on;
if length(p_radial) > 1
    theory = p_radial(1) * sqrt(r(1)) ./ sqrt(r);
    plot(r * DX * 1e3, theory, 'r--', 'LineWidth', 1.5, 'DisplayName', '1/\surd r theory');
end
xlabel('r [mm]');
ylabel('|p| [Pa]');
title('Amplitude decay verification (2D)');
legend('Location', 'best');
grid on;
outPath = fullfile(figDir, 'step1_decay_check.png');
print(gcf, outPath, '-dpng', '-r150');
fprintf('Saved figure: %s\n', outPath);

%% Summary
fprintf('\n%s\n', repmat('=', 1, 60));
fprintf('STEP 1 COMPLETE\n');
fprintf('%s\n', repmat('=', 1, 60));
fprintf('  Grid:           %d x %d\n', NX, NY);
fprintf('  dx:             %.1f µm\n', DX*1e6);
fprintf('  dt:             %.3f ns\n', DT*1e9);
fprintf('  CFL:            %.3f  (< %.3f ✓)\n', CFL, 1/sqrt(2));
fprintf('  Boundary:       Neumann (∂p/∂n = 0) on all edges\n');
fprintf('  Boundary cells: %d\n', sum(bc_mask(:) == 1));
fprintf('  Interior cells: %d\n', sum(bc_mask(:) == 0));
fprintf('  Figures saved:  %s/\n', figDir);
fprintf('%s\n', repmat('=', 1, 60));

%% Local functions
function val = source_pulse(t, amp, delay, width)
    val = amp * exp(-((t - delay) / width)^2);
end

function lap = compute_laplacian(p, mask)
    [nx, ny] = size(p);
    lap = zeros(size(p));
    
    % LEFT neighbor (i-1)
    left = zeros(size(p));
    left(2:end, :) = p(1:end-1, :);  % interior neighbors
    % Neumann (mask=1): p_ghost = p(1, :) (reflect)
    neumann_left = (mask(1, :) == 1);
    left(1, neumann_left) = p(1, neumann_left);
    
    % RIGHT neighbor (i+1)
    right = zeros(size(p));
    right(1:end-1, :) = p(2:end, :);  % interior neighbors
    neumann_right = (mask(end, :) == 1);
    right(end, neumann_right) = p(end, neumann_right);
    
    % BOTTOM neighbor (j-1)
    bottom = zeros(size(p));
    bottom(:, 2:end) = p(:, 1:end-1);  % interior neighbors
    neumann_bottom = (mask(:, 1) == 1);
    bottom(neumann_bottom, 1) = p(neumann_bottom, 1);
    
    % TOP neighbor (j+1)
    top = zeros(size(p));
    top(:, 1:end-1) = p(:, 2:end);  % interior neighbors
    neumann_top = (mask(:, end) == 1);
    top(neumann_top, end) = p(neumann_top, end);
    
    % Assemble Laplacian
    lap = left + right + bottom + top - 4 * p;
end

function cmap = rdBu_r(n)
    if nargin < 1, n = 256; end
    half = floor(n/2);
    % Red (negative) to white
    r1 = linspace(0.4039, 1, half)';
    g1 = linspace(0.0000, 1, half)';
    b1 = linspace(0.1216, 1, half)';
    % White to blue (positive)
    r2 = linspace(1, 0.1686, n-half)';
    g2 = linspace(1, 0.2980, n-half)';
    b2 = linspace(1, 0.5373, n-half)';
    cmap = [r1 g1 b1; r2 g2 b2];
end
