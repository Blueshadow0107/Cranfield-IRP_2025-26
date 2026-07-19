%% Y-Junction Acoustic Logic Unit — 2D FDTD Step 4
% Phase sweep demonstration.
%
% For each phase difference φ between the two inlet sources:
%     - Reset fields to zero
%     - Run to steady state (1700 steps)
%     - Measure outlet pressure over last 3 periods
%
% Outputs:
%     - Amplitude vs phase curve (3 metrics)
%     - Side-by-side pressure fields for φ = 0° and 180°

clear; clc; close all;

%% Reuse Step 3 parameters and geometry
DX = 5e-5;
C = 1500.0;
FREQ = 1.0e6;
LAMBDA = C / FREQ;
CFL = 0.5;
DT = CFL * DX / (C * sqrt(2));
coef = (C * DT / DX)^2;

LAMBDA_CELLS = round(LAMBDA / DX);
W_CHANNEL = round(0.3 * LAMBDA_CELLS);
NX = 250;
NY = 200;
L_SECTION = floor(NX / 3);
X_BEND = L_SECTION;
X_MERGE = 2 * L_SECTION;
Y_MERGE = round(NY / 2);
INLET_ANGLE_RAD = deg2rad(30);
Y_OFFSET = L_SECTION * tan(INLET_ANGLE_RAD);
Y_INLET_A = Y_MERGE + Y_OFFSET;
Y_INLET_B = Y_MERGE - Y_OFFSET;

SRC_AMP = 1.0;
STEPS_PER_PERIOD = round(1.0 / FREQ / DT);
NT = 1700;                           % ~20 periods to reach steady state
MEASURE_STEPS = 3 * STEPS_PER_PERIOD; % Last 3 periods for measurement

fprintf('Steps per period = %d\n', STEPS_PER_PERIOD);
fprintf('Total steps per run = %d\n', NT);
fprintf('Measurement window = last %d steps (%.1f periods)\n', MEASURE_STEPS, MEASURE_STEPS/STEPS_PER_PERIOD);

% Sources and probe
srcA_x = 1;
srcA_y = round(Y_INLET_A);
srcB_x = 1;
srcB_y = round(Y_INLET_B);
PROBE_X = X_MERGE + LAMBDA_CELLS;
PROBE_Y = Y_MERGE;

%% Geometry builder (copy from Step 3)
bc_mask = build_y_junction_mask(NX, NY, X_BEND, X_MERGE, Y_MERGE, W_CHANNEL, Y_INLET_A, Y_INLET_B);

%% Phase sweep
PHASE_DEGREES = 0:30:360;   % 0°, 30°, ..., 360°
PHASE_RADIANS = deg2rad(PHASE_DEGREES);

results_peak = zeros(size(PHASE_DEGREES));
results_rms = zeros(size(PHASE_DEGREES));
results_pkpk = zeros(size(PHASE_DEGREES));
snapshot_fields = struct();   % Store fields for selected phases

fprintf('\nRunning phase sweep: %d cases\n', length(PHASE_DEGREES));
fprintf('%s\n', repmat('=', 1, 50));

for idx = 1:length(PHASE_DEGREES)
    phi_deg = PHASE_DEGREES(idx);
    phi_rad = PHASE_RADIANS(idx);
    
    [probe, field] = run_fdtd(phi_rad, NT, NX, NY, srcA_x, srcA_y, srcB_x, srcB_y, PROBE_X, PROBE_Y, SRC_AMP, FREQ, DT, coef, bc_mask);
    
    % Extract measurement window (last 3 periods)
    window = probe(end-MEASURE_STEPS+1:end);
    
    % Three metrics
    peak = max(abs(window));
    rms_val = sqrt(mean(window.^2));
    pkpk = max(window) - min(window);
    
    results_peak(idx) = peak;
    results_rms(idx) = rms_val;
    results_pkpk(idx) = pkpk;
    
    % Store field for selected phases
    if ismember(phi_deg, [0, 90, 180, 270])
        field_name = sprintf('phi%d', phi_deg);
        snapshot_fields.(field_name) = field;
    end
    
    fprintf('  φ = %3d°  peak = %.4f  RMS = %.4f  Pk-Pk = %.4f\n', phi_deg, peak, rms_val, pkpk);
end
fprintf('%s\n', repmat('=', 1, 50));

%% Visualisation
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

% --- Figure 1: Amplitude vs Phase ---
figure('Position', [100 100 800 450], 'Color', 'w');
plot(PHASE_DEGREES, results_peak, 'o-', 'LineWidth', 1.5, 'DisplayName', 'Peak |p|', 'Color', [0 0.4470 0.7410]);
hold on;
plot(PHASE_DEGREES, results_rms, 's-', 'LineWidth', 1.5, 'DisplayName', 'RMS p', 'Color', [0.8500 0.3250 0.0980]);
plot(PHASE_DEGREES, results_pkpk, '^-', 'LineWidth', 1.5, 'DisplayName', 'Peak-to-peak', 'Color', [0.9290 0.6940 0.1250]);
xline(0, 'Color', [0.5 0.5 0.5], 'LineStyle', '--', 'Alpha', 0.3);
xline(180, 'Color', [0.5 0.5 0.5], 'LineStyle', '--', 'Alpha', 0.3);
ylim_vals = ylim;
text(0, ylim_vals(2)*0.95, 'Constructive\n(AND/OR)', 'HorizontalAlignment', 'center', 'FontSize', 9);
text(180, ylim_vals(2)*0.95, 'Destructive\n(XOR)', 'HorizontalAlignment', 'center', 'FontSize', 9);
xlabel('Phase difference φ [degrees]');
ylabel('Outlet pressure [Pa]');
title('Logic Gate Performance: Outlet Pressure vs Input Phase');
xticks(PHASE_DEGREES);
legend('Location', 'best');
grid on;
outPath = fullfile(figDir, 'step4_phase_sweep.png');
print(gcf, outPath, '-dpng', '-r150');
fprintf('\nSaved figure: %s\n', outPath);

% --- Figure 2: Side-by-side pressure fields ---
field_names = fieldnames(snapshot_fields);
n_fields = length(field_names);
ncols = 2;
nrows = ceil(n_fields / ncols);
figure('Position', [100 100 500*ncols 450*nrows], 'Color', 'w');

vmax = 0;
for k = 1:n_fields
    vmax = max(vmax, max(abs(snapshot_fields.(field_names{k})(:))));
end

x = linspace(0, NX*DX*1e3, NX);
y = linspace(0, NY*DX*1e3, NY);

for k = 1:n_fields
    subplot(nrows, ncols, k);
    field = snapshot_fields.(field_names{k});
    imagesc(x, y, field.');
    axis image;
    set(gca, 'YDir', 'normal');
    colormap(gca, rdBu_r(256));
    caxis([-vmax vmax]);
    colorbar;
    hold on;
    contour(x, y, bc_mask.', [0.5 0.5], 'k-', 'LineWidth', 0.5);
    % Extract phase value from field name (e.g., 'phi90' -> 90)
    phi_val = sscanf(field_names{k}, 'phi%d');
    title(sprintf('φ = %d°', phi_val));
    xlabel('x [mm]');
    ylabel('y [mm]');
end

sgtitle('Step 4: Pressure Field at Selected Phase Differences', 'FontSize', 14);
outPath = fullfile(figDir, 'step4_snapshots.png');
print(gcf, outPath, '-dpng', '-r150');
fprintf('Saved figure: %s\n', outPath);

% --- Figure 3: Logic contrast summary ---
logic_high = results_rms(1);       % φ = 0°
logic_low = results_rms(7);        % φ = 180° (index 7 = 0:30:360 -> 180 is the 7th element)
contrast = (logic_high - logic_low) / (logic_high + logic_low);

figure('Position', [100 100 500 350], 'Color', 'w');
bar([logic_high, logic_low], 'FaceColor', 'flat');
set(gca, 'XTickLabel', {'Logic "1" (φ = 0°)', 'Logic "0" (φ = 180°)'});
hold on;
yline((logic_high + logic_low)/2, 'k--', 'LineWidth', 1.5, 'Label', sprintf('Threshold = %.4f Pa', (logic_high+logic_low)/2));
ylabel('RMS pressure [Pa]');
title(sprintf('Logic Contrast Ratio = %.3f', contrast));
outPath = fullfile(figDir, 'step4_logic_contrast.png');
print(gcf, outPath, '-dpng', '-r150');
fprintf('Saved figure: %s\n', outPath);

%% Summary
fprintf('\n%s\n', repmat('=', 1, 50));
fprintf('STEP 4 COMPLETE\n');
fprintf('%s\n', repmat('=', 1, 50));
fprintf('Logic HIGH (φ = 0°):   RMS = %.4f Pa\n', logic_high);
fprintf('Logic LOW  (φ = 180°): RMS = %.4f Pa\n', logic_low);
fprintf('Contrast ratio:        %.3f\n', contrast);
fprintf('Threshold:             %.4f Pa\n', (logic_high + logic_low)/2);
fprintf('%s\n', repmat('=', 1, 50));

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
    left(2:end, :) = p(1:end-1, :);
    wall_left = false(size(mask));
    wall_left(2:end, :) = (mask(1:end-1, :) == 1);
    left(wall_left) = p(wall_left);
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

function [probe_trace, final_field] = run_fdtd(phase_b, NT, NX, NY, srcA_x, srcA_y, srcB_x, srcB_y, PROBE_X, PROBE_Y, SRC_AMP, FREQ, DT, coef, bc_mask)
    p_prev = zeros(NX, NY);
    p_curr = zeros(NX, NY);
    p_next = zeros(NX, NY);
    probe_trace = zeros(1, NT);
    
    for n = 0:NT-1
        t = n * DT;
        p_curr(srcA_x, srcA_y) = SRC_AMP * sin(2 * pi * FREQ * t);
        p_curr(srcB_x, srcB_y) = SRC_AMP * sin(2 * pi * FREQ * t + phase_b);
        
        laplacian = compute_laplacian(p_curr, bc_mask);
        p_next = 2 * p_curr - p_prev + coef * laplacian;
        p_next(bc_mask == 1) = p_curr(bc_mask == 1);
        
        % Swap
        temp = p_prev;
        p_prev = p_curr;
        p_curr = p_next;
        p_next = temp;
        
        probe_trace(n+1) = p_curr(PROBE_X, PROBE_Y);
    end
    
    final_field = p_curr;
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
