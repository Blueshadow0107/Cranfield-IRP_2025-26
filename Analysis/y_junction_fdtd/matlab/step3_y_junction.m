%% Y-Junction Acoustic Logic Unit — 2D FDTD Step 3
% Three-section geometry:
%     1. Two straight horizontal inlets from left edge
%     2. Angled merge region (±30°)
%     3. Straight outlet to right edge
%
% All three sections have equal horizontal extent (~83 cells).
%
% Fixes applied:
%     - Internal walls reflect in Laplacian (ghost-cell Neumann)
%     - Wall cells frozen after each time step
%     - Hard source overwrite (no accumulation)
%
% Grid: dx = 50 µm, c = 1500 m/s, f = 1 MHz → λ = 1.5 mm = 30 cells

clear; clc; close all;

%% Physical and numerical parameters
DX = 5e-5;                    % Grid spacing [m] = 50 µm
C = 1500.0;                   % Wave speed [m/s] (water)
FREQ = 1.0e6;                 % Source frequency [Hz]
LAMBDA = C / FREQ;            % Wavelength [m]

CFL = 0.5;                    % CFL safety factor
DT = CFL * DX / (C * sqrt(2));
coef = (C * DT / DX)^2;

% Stability check
dt_max = DX / (C * sqrt(2));
assert(DT <= dt_max, 'CFL violation: DT=%.3e > dt_max=%.3e', DT, dt_max);

%% Geometry parameters (in cells)
LAMBDA_CELLS = round(LAMBDA / DX);    % Should be 30
W_CHANNEL = round(0.3 * LAMBDA_CELLS);   % 0.3λ ≈ 9 cells

% Domain size
NX = 250;                     % 12.5 mm
NY = 200;                     % 10 mm

% Three sections with equal horizontal extent
L_SECTION = floor(NX / 3);    % ~83 cells each
X_BEND = L_SECTION;           % Where horizontal inlets become angled
X_MERGE = 2 * L_SECTION;      % Where angled inlets meet / outlet begins
Y_MERGE = round(NY / 2);      % Domain centre vertically

% Angled section geometry
INLET_ANGLE_DEG = 30;
INLET_ANGLE_RAD = deg2rad(INLET_ANGLE_DEG);
L_ANGLED = L_SECTION / cos(INLET_ANGLE_RAD);  % Actual segment length
Y_OFFSET = L_SECTION * tan(INLET_ANGLE_RAD);   % Vertical drop/rise

% Inlet centerline y-positions
Y_INLET_A = Y_MERGE + Y_OFFSET;
Y_INLET_B = Y_MERGE - Y_OFFSET;

fprintf('%s\n', repmat('=', 1, 60));
fprintf('STEP 3: Y-Junction Geometry Setup\n');
fprintf('%s\n', repmat('=', 1, 60));
fprintf('dx = %.1f µm,  dt = %.3f ns,  c = %.0f m/s\n', DX*1e6, DT*1e9, C);
fprintf('Frequency = %.2f MHz,  λ = %.2f mm = %d cells\n', FREQ/1e6, LAMBDA*1e3, LAMBDA_CELLS);
fprintf('CFL = %.3f  (stability limit = %.3f)\n', CFL, 1/sqrt(2));
fprintf('Domain: %d x %d = %.2f mm x %.2f mm\n', NX, NY, NX*DX*1e3, NY*DX*1e3);
fprintf('Channel width = 0.3λ = %d cells = %.0f µm\n', W_CHANNEL, W_CHANNEL*DX*1e6);
fprintf('Section length (horizontal) = %d cells = %.2f mm\n', L_SECTION, L_SECTION*DX*1e3);
fprintf('Merge point = (%d, %d)\n', X_MERGE, Y_MERGE);
fprintf('Inlet A center y = %.1f\n', Y_INLET_A);
fprintf('Inlet B center y = %.1f\n', Y_INLET_B);
fprintf('%s\n', repmat('=', 1, 60));

%% Geometry builder: distance-to-segment
bc_mask = build_y_junction_mask(NX, NY, X_BEND, X_MERGE, Y_MERGE, W_CHANNEL, Y_INLET_A, Y_INLET_B);

fprintf('Interior cells: %d\n', sum(bc_mask(:) == 0));
fprintf('Wall cells:     %d\n', sum(bc_mask(:) == 1));

%% Visualise geometry
scriptPath = mfilename('fullpath');
if isempty(scriptPath)
    figDir = fullfile(pwd, 'figures');
else
    [scriptDir, ~, ~] = fileparts(scriptPath);
    figDir = fullfile(scriptDir, '..', 'figures');
end
if ~exist(figDir, 'dir')
    mkdir(figDir);
end

figure('Position', [100 100 800 500], 'Color', 'w');
imagesc([0 NX*DX*1e3], [0 NY*DX*1e3], bc_mask.');
axis image;
set(gca, 'YDir', 'normal');
colormap(gca, gray);
hold on;
plot(X_MERGE*DX*1e3, Y_MERGE*DX*1e3, 'ro', 'MarkerSize', 8, 'DisplayName', 'Merge point');
xline(X_BEND*DX*1e3, 'g--', 'Alpha', 0.5, 'Label', 'Bend start');
xline(X_MERGE*DX*1e3, 'b--', 'Alpha', 0.5, 'Label', 'Merge start');
title('Y-Junction Geometry (Step 3)');
xlabel('x [mm]');
ylabel('y [mm]');
legend('Location', 'best');
outPath = fullfile(figDir, 'step3_geometry.png');
print(gcf, outPath, '-dpng', '-r150');
fprintf('Saved geometry figure: %s\n', outPath);

%% FDTD field arrays and source parameters
p_prev = zeros(NX, NY);
p_curr = zeros(NX, NY);
p_next = zeros(NX, NY);

SRC_AMP = 1.0;
PHASE_A = 0.0;
PHASE_B = 0.0;

% Sources at left edge of horizontal inlets
srcA_x = 1;  % Python 0 → MATLAB 1
srcA_y = round(Y_INLET_A);
srcB_x = 1;
srcB_y = round(Y_INLET_B);

% Probe in outlet, 1λ downstream of merge
PROBE_X = X_MERGE + LAMBDA_CELLS;
PROBE_Y = Y_MERGE;

fprintf('Source A at (%d, %d)\n', srcA_x, srcA_y);
fprintf('Source B at (%d, %d)\n', srcB_x, srcB_y);
fprintf('Probe at (%d, %d)\n', PROBE_X, PROBE_Y);

%% Time stepping
NT = 800;
T_PERIOD = 1.0 / FREQ;
STEPS_PER_PERIOD = round(T_PERIOD / DT);

fprintf('Steps per period = %d\n', STEPS_PER_PERIOD);

snapshot_interval = STEPS_PER_PERIOD;
snapshots = cell(1, ceil(NT/snapshot_interval) + 1);
times = zeros(1, ceil(NT/snapshot_interval) + 1);
probe_trace = zeros(1, NT);
snap_idx = 0;

fprintf('\nRunning %d time steps...\n', NT);
for n = 0:NT-1
    t = n * DT;
    
    % Inject hard sinusoidal sources (Fix 3: overwrite, not accumulate)
    p_curr(srcA_x, srcA_y) = SRC_AMP * sin(2 * pi * FREQ * t + PHASE_A);
    p_curr(srcB_x, srcB_y) = SRC_AMP * sin(2 * pi * FREQ * t + PHASE_B);
    
    % Compute Laplacian
    laplacian = compute_laplacian(p_curr, bc_mask);
    
    % Time update (leapfrog)
    p_next = 2 * p_curr - p_prev + coef * laplacian;
    
    % Freeze wall cells (Fix 2)
    p_next(bc_mask == 1) = p_curr(bc_mask == 1);
    
    % Swap
    temp = p_prev;
    p_prev = p_curr;
    p_curr = p_next;
    p_next = temp;
    
    % Record probe
    probe_trace(n+1) = p_curr(PROBE_X, PROBE_Y);
    
    % Save snapshot
    if mod(n, snapshot_interval) == 0
        snap_idx = snap_idx + 1;
        snapshots{snap_idx} = p_curr;
        times(snap_idx) = t;
    end
    
    if mod(n, 2 * STEPS_PER_PERIOD) == 0
        fprintf('  Step %4d/%d  t = %.2f µs  max |p| = %.4f\n', n, NT, t*1e6, max(abs(p_curr(:))));
    end
end
fprintf('Done.\n\n');

snapshots = snapshots(1:snap_idx);
times = times(1:snap_idx);

%% Visualisation
n_snaps = length(snapshots);
ncols = 5;
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
    hold on;
    contour(x, y, bc_mask.', [0.5 0.5], 'k-', 'LineWidth', 0.5);
    title(sprintf('t = %.1f µs (%.1f T)', times(i)*1e6, times(i)/T_PERIOD));
    xlabel('x [mm]');
    ylabel('y [mm]');
end

sgtitle('Step 3: Y-Junction Wave Propagation (φ = 0°)', 'FontSize', 14);
outPath = fullfile(figDir, 'step3_y_junction.png');
print(gcf, outPath, '-dpng', '-r150');
fprintf('Saved figure: %s\n', outPath);

% Probe trace
figure('Position', [100 100 800 350], 'Color', 'w');
t_probe = (0:length(probe_trace)-1) * DT * 1e6;
plot(t_probe, probe_trace, 'b-', 'LineWidth', 1.0);
xlabel('t [µs]');
ylabel('p [Pa]');
title(sprintf('Probe at outlet (%.2f mm, %.2f mm)', PROBE_X*DX*1e3, PROBE_Y*DX*1e3));
grid on;
outPath = fullfile(figDir, 'step3_probe.png');
print(gcf, outPath, '-dpng', '-r150');
fprintf('Saved figure: %s\n', outPath);

%% Summary
fprintf('\n%s\n', repmat('=', 1, 60));
fprintf('STEP 3 COMPLETE\n');
fprintf('%s\n', repmat('=', 1, 60));
fprintf('  Grid:           %d x %d\n', NX, NY);
fprintf('  dx:             %.1f µm\n', DX*1e6);
fprintf('  dt:             %.3f ns\n', DT*1e9);
fprintf('  Frequency:      %.2f MHz\n', FREQ/1e6);
fprintf('  Wavelength:     %.2f mm = %d cells\n', LAMBDA*1e3, LAMBDA_CELLS);
fprintf('  Channel width:  %d cells = %.0f µm\n', W_CHANNEL, W_CHANNEL*DX*1e6);
fprintf('  Section length: %d cells = %.2f mm\n', L_SECTION, L_SECTION*DX*1e3);
fprintf('  Merge angle:    ±%d° (60° total)\n', INLET_ANGLE_DEG);
fprintf('  Phase A:        %.2f rad\n', PHASE_A);
fprintf('  Phase B:        %.2f rad\n', PHASE_B);
fprintf('  Figures saved:  %s/\n', figDir);
fprintf('%s\n', repmat('=', 1, 60));

%% Local functions
function d = point_segment_distance(px, py, x1, y1, x2, y2)
    dx_seg = x2 - x1;
    dy_seg = y2 - y1;
    if dx_seg == 0 && dy_seg == 0
        d = hypot(px - x1, py - y1);
        return;
    end
    t = max(0.0, min(1.0, ((px - x1) * dx_seg + (py - y1) * dy_seg) / (dx_seg^2 + dy_seg^2)));
    proj_x = x1 + t * dx_seg;
    proj_y = y1 + t * dy_seg;
    d = hypot(px - proj_x, py - proj_y);
end

function mask = build_y_junction_mask(nx, ny, x_bend, x_merge, y_merge, w, y_inlet_a, y_inlet_b)
    mask = ones(nx, ny, 'int32');
    
    % 5 segments: 2 horizontal inlets, 2 angled inlets, 1 outlet
    % Coordinates are 0-based (Python style), loops use 1-based MATLAB indexing
    segments = {
        [0, y_inlet_a, x_bend, y_inlet_a];
        [x_bend, y_inlet_a, x_merge, y_merge];
        [0, y_inlet_b, x_bend, y_inlet_b];
        [x_bend, y_inlet_b, x_merge, y_merge];
        [x_merge, y_merge, nx - 1, y_merge];
    };
    
    for i = 1:nx
        for j = 1:ny
            d_min = inf;
            for s = 1:length(segments)
                seg = segments{s};
                d = point_segment_distance(i-1, j-1, seg(1), seg(2), seg(3), seg(4));
                if d < d_min
                    d_min = d;
                end
            end
            if d_min < w / 2
                mask(i, j) = 0;
            end
        end
    end
end

function lap = compute_laplacian(p, mask)
    % LEFT neighbor (i-1)
    left = zeros(size(p));
    left(2:end, :) = p(1:end-1, :);  % interior neighbors
    wall_left = false(size(mask));
    wall_left(2:end, :) = (mask(1:end-1, :) == 1);
    left(wall_left) = p(wall_left);
    % Boundary at i=1
    neumann_left = (mask(1, :) == 1);
    left(1, neumann_left) = p(1, neumann_left);
    
    % RIGHT neighbor (i+1)
    right = zeros(size(p));
    right(1:end-1, :) = p(2:end, :);
    wall_right = false(size(mask));
    wall_right(1:end-1, :) = (mask(2:end, :) == 1);
    right(wall_right) = p(wall_right);
    neumann_right = (mask(end, :) == 1);
    right(end, neumann_right) = p(end, neumann_right);
    
    % BOTTOM neighbor (j-1)
    bottom = zeros(size(p));
    bottom(:, 2:end) = p(:, 1:end-1);
    wall_bottom = false(size(mask));
    wall_bottom(:, 2:end) = (mask(:, 1:end-1) == 1);
    bottom(wall_bottom) = p(wall_bottom);
    neumann_bottom = (mask(:, 1) == 1);
    bottom(neumann_bottom, 1) = p(neumann_bottom, 1);
    
    % TOP neighbor (j+1)
    top = zeros(size(p));
    top(:, 1:end-1) = p(:, 2:end);
    wall_top = false(size(mask));
    wall_top(:, 1:end-1) = (mask(:, 2:end) == 1);
    top(wall_top) = p(wall_top);
    neumann_top = (mask(:, end) == 1);
    top(neumann_top, end) = p(neumann_top, end);
    
    lap = left + right + bottom + top - 4 * p;
end

function cmap = rdBu_r(n)
    if nargin < 1, n = 256; end
    half = floor(n/2);
    r1 = linspace(0.4039, 1, half)';
    g1 = linspace(0.0000, 1, half)';
    b1 = linspace(0.1216, 1, half)';
    r2 = linspace(1, 0.1686, n-half)';
    g2 = linspace(1, 0.2980, n-half)';
    b2 = linspace(1, 0.5373, n-half)';
    cmap = [r1 g1 b1; r2 g2 b2];
end
