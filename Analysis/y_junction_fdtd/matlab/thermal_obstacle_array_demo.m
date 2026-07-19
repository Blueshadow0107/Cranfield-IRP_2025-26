%% Thermal Obstacle Array — Velocity Readout Demo
% Single source, 3 circular obstacles, 2 heated regions, 2 output probes.
% Sweeps T1 (pre-obstacle) and T2 (post-obstacle) to map thermal routing.
%
% Physics: spatially varying c(x,y) from temperature fields.
% Velocity computed from momentum equation: dv/dt = -(1/rho) grad(p)
% Density kept constant; only sound speed varies with temperature.

clc; clear; close all;

%% ============================================================
% 1. Physical and numerical parameters
%% ============================================================
DX = 5e-5;                    % Grid spacing [m] = 50 um
C0 = 1500.0;                  % Reference sound speed [m/s] (water at 20 C)
RHO0 = 998.0;                 % Density [kg/m^3] (constant)
FREQ = 1.0e6;                 % Source frequency [Hz]
LAMBDA = C0 / FREQ;           % Wavelength [m] = 1.5 mm
LAMBDA_CELLS = round(LAMBDA / DX);   % = 30 cells

CFL = 0.5;
DT = CFL * DX / (C0 * sqrt(2));
coef0 = (C0 * DT / DX)^2;     % Reference coefficient

% Stability check
dt_max = DX / (C0 * sqrt(2));
assert(DT <= dt_max, 'CFL violation');

% Thermal coefficient for water: c decreases ~0.24%% per degC
ALPHA_C = 2.4e-3;             % [1/degC]

% Domain size
NX = 200;
NY = 150;

% Simulation duration: run to steady state + measurement window
PERIOD = 1 / FREQ;
STEPS_PER_PERIOD = round(PERIOD / DT);
NT_WARMUP = 20 * STEPS_PER_PERIOD;   % 20 periods to reach steady state
NT_MEASURE = 3 * STEPS_PER_PERIOD;   % Measure over last 3 periods
NT = NT_WARMUP + NT_MEASURE;

%% ============================================================
% 2. Geometry: obstacles, sources, probes, heated regions
%% ============================================================
% Obstacles: 3 circles, diameter ~lambda/2, diagonal arrangement
obs_centers = [70, 60;
               100, 75;
               130, 90];
obs_radius = round(LAMBDA_CELLS / 4);   % ~7-8 cells

obs_mask = false(NX, NY);
for k = 1:size(obs_centers, 1)
    cx = obs_centers(k, 1);
    cy = obs_centers(k, 2);
    for i = 1:NX
        for j = 1:NY
            if sqrt((i-cx)^2 + (j-cy)^2) <= obs_radius
                obs_mask(i, j) = true;
            end
        end
    end
end

% Source: left edge, centre height
SRC_X = 5;
SRC_Y = round(NY / 2);
SRC_AMP = 1.0;

% Probes: right edge, upper and lower
PROBE1 = [NX-5, round(NY/3)];       % upper
PROBE2 = [NX-5, round(2*NY/3)];     % lower

% Heated regions: rectangular strips
% Region 1: before first obstacle
reg1_x = 35:55;
reg1_y = 45:105;

% Region 2: after last obstacle
reg2_x = 145:165;
reg2_y = 45:105;

%% ============================================================
% 3. Helper function: build c-field for given temperatures
%% ============================================================
function c_field = build_c_field(T1, T2, C0, ALPHA_C, NX, NY, reg1_x, reg1_y, reg2_x, reg2_y)
    c_field = C0 * ones(NX, NY);
    c_field(reg1_x, reg1_y) = C0 * (1 - ALPHA_C * T1);
    c_field(reg2_x, reg2_y) = C0 * (1 - ALPHA_C * T2);
end

%% ============================================================
% 4. Helper function: run FDTD with thermal field
%% ============================================================
function [v1_rms, v2_rms, p_final, vx_final, vy_final] = ...
    run_fdtd_thermal(c_field, obs_mask, SRC_X, SRC_Y, PROBE1, PROBE2, ...
                     SRC_AMP, FREQ, DT, DX, RHO0, NT, NT_WARMUP, NT_MEASURE)

    [NX, NY] = size(c_field);
    
    % Coefficient field for pressure update
    coef_field = (c_field * DT / DX).^2;
    
    % Field arrays
    p_prev = zeros(NX, NY);
    p_curr = zeros(NX, NY);
    p_next = zeros(NX, NY);
    vx = zeros(NX, NY);
    vy = zeros(NX, NY);
    
    % Buffers for probe velocity history during measurement window
    v1_hist = zeros(NT_MEASURE, 1);
    v2_hist = zeros(NT_MEASURE, 1);
    hist_idx = 0;
    
    for n = 1:NT
        t = (n - 1) * DT;
        
        % Hard source overwrite
        p_curr(SRC_X, SRC_Y) = SRC_AMP * sin(2 * pi * FREQ * t);
        
        % Laplacian via circshift (periodic; boundaries handled below)
        laplacian = (circshift(p_curr, [1, 0]) + circshift(p_curr, [-1, 0]) ...
                   + circshift(p_curr, [0, 1]) + circshift(p_curr, [0, -1]) ...
                   - 4 * p_curr) / DX^2;
        
        % Pressure update with spatially varying c
        p_next = 2 * p_curr - p_prev + coef_field .* laplacian;
        
        % Velocity update from momentum equation
        % grad(p) computed with central differences
        grad_px = (circshift(p_curr, [0, -1]) - circshift(p_curr, [0, 1])) / (2 * DX);
        grad_py = (circshift(p_curr, [-1, 0]) - circshift(p_curr, [1, 0])) / (2 * DX);
        vx = vx - (DT / RHO0) * grad_px;
        vy = vy - (DT / RHO0) * grad_py;
        
        % Neumann boundary conditions (hard walls)
        p_next(1, :) = p_next(2, :);           % left
        p_next(end, :) = p_next(end-1, :);     % right
        p_next(:, 1) = p_next(:, 2);           % bottom
        p_next(:, end) = p_next(:, end-1);     % top
        
        % Freeze obstacle cells (hard walls)
        p_next(obs_mask) = p_curr(obs_mask);
        vx(obs_mask) = 0;
        vy(obs_mask) = 0;
        
        % Zero velocity at domain boundaries (no slip)
        vx(1, :) = 0; vx(end, :) = 0; vx(:, 1) = 0; vx(:, end) = 0;
        vy(1, :) = 0; vy(end, :) = 0; vy(:, 1) = 0; vy(:, end) = 0;
        
        % Cycle buffers
        temp = p_prev;
        p_prev = p_curr;
        p_curr = p_next;
        p_next = temp;
        
        % Record probe velocities during measurement window
        if n > NT_WARMUP
            hist_idx = hist_idx + 1;
            v1_hist(hist_idx) = sqrt(vx(PROBE1(1), PROBE1(2))^2 + vy(PROBE1(1), PROBE1(2))^2);
            v2_hist(hist_idx) = sqrt(vx(PROBE2(1), PROBE2(2))^2 + vy(PROBE2(1), PROBE2(2))^2);
        end
    end
    
    % RMS velocity over measurement window
    v1_rms = sqrt(mean(v1_hist.^2));
    v2_rms = sqrt(mean(v2_hist.^2));
    
    % Final fields for visualization
    p_final = p_curr;
    vx_final = vx;
    vy_final = vy;
end

%% ============================================================
% 5. Parameter sweep
%% ============================================================
T1_vals = 0:10:50;   % degC above ambient
T2_vals = 0:10:50;
N1 = length(T1_vals);
N2 = length(T2_vals);

v1_map = zeros(N1, N2);
v2_map = zeros(N1, N2);
ratio_map = zeros(N1, N2);

fprintf('Running parameter sweep: %d configurations...\n', N1 * N2);
tic;

for i = 1:N1
    for j = 1:N2
        T1 = T1_vals(i);
        T2 = T2_vals(j);
        
        c_field = build_c_field(T1, T2, C0, ALPHA_C, NX, NY, reg1_x, reg1_y, reg2_x, reg2_y);
        
        [v1_rms, v2_rms, ~, ~, ~] = run_fdtd_thermal(...
            c_field, obs_mask, SRC_X, SRC_Y, PROBE1, PROBE2, ...
            SRC_AMP, FREQ, DT, DX, RHO0, NT, NT_WARMUP, NT_MEASURE);
        
        v1_map(i, j) = v1_rms;
        v2_map(i, j) = v2_rms;
        ratio_map(i, j) = v1_rms / (v2_rms + eps);
        
        fprintf('  T1=%2d, T2=%2d -> v1=%.4f, v2=%.4f, ratio=%.3f\n', ...
            T1, T2, v1_rms, v2_rms, ratio_map(i, j));
    end
end

elapsed = toc;
fprintf('Sweep complete in %.1f seconds.\n', elapsed);

%% ============================================================
% 6. Visualisation
%% ============================================================

% 6a. Run one case for field visualisation (T1=20, T2=30)
T1_demo = 20;
T2_demo = 30;
c_field_demo = build_c_field(T1_demo, T2_demo, C0, ALPHA_C, NX, NY, reg1_x, reg1_y, reg2_x, reg2_y);
[~, ~, p_demo, vx_demo, vy_demo] = run_fdtd_thermal(...
    c_field_demo, obs_mask, SRC_X, SRC_Y, PROBE1, PROBE2, ...
    SRC_AMP, FREQ, DT, DX, RHO0, NT, NT_WARMUP, NT_MEASURE);

figure('Position', [100 100 1400 500]);

% Pressure field
subplot(1, 3, 1);
imagesc(p_demo'); axis image; colormap(gca, jet); colorbar;
hold on;
contour(obs_mask', 1, 'w-', 'LineWidth', 1.5);
rectangle('Position', [reg1_x(1), reg1_y(1), length(reg1_x), length(reg1_y)], ...
    'EdgeColor', 'r', 'LineStyle', '--', 'LineWidth', 1.5);
rectangle('Position', [reg2_x(1), reg2_y(1), length(reg2_x), length(reg2_y)], ...
    'EdgeColor', 'r', 'LineStyle', '--', 'LineWidth', 1.5);
plot(SRC_X, SRC_Y, 'g*', 'MarkerSize', 15, 'LineWidth', 2);
plot(PROBE1(1), PROBE1(2), 'mo', 'MarkerSize', 10, 'LineWidth', 2);
plot(PROBE2(1), PROBE2(2), 'co', 'MarkerSize', 10, 'LineWidth', 2);
hold off;
title(sprintf('Pressure field (T1=%dC, T2=%dC)', T1_demo, T2_demo));
xlabel('x'); ylabel('y');

% Velocity magnitude
subplot(1, 3, 2);
v_mag = sqrt(vx_demo.^2 + vy_demo.^2);
imagesc(v_mag'); axis image; colormap(gca, hot); colorbar;
hold on;
contour(obs_mask', 1, 'w-', 'LineWidth', 1.5);
plot(SRC_X, SRC_Y, 'g*', 'MarkerSize', 15, 'LineWidth', 2);
plot(PROBE1(1), PROBE1(2), 'mo', 'MarkerSize', 10, 'LineWidth', 2);
plot(PROBE2(1), PROBE2(2), 'co', 'MarkerSize', 10, 'LineWidth', 2);
hold off;
title('Velocity magnitude |v|');
xlabel('x'); ylabel('y');

% Velocity vector field (subsampled)
subplot(1, 3, 3);
skip = 6;
[X, Y] = meshgrid(1:NX, 1:NY);
quiver(X(1:skip:end, 1:skip:end)', Y(1:skip:end, 1:skip:end)', ...
    vx_demo(1:skip:end, 1:skip:end)', vy_demo(1:skip:end, 1:skip:end)', ...
    2, 'b');
hold on;
contour(obs_mask', 1, 'k-', 'LineWidth', 1.5);
plot(SRC_X, SRC_Y, 'g*', 'MarkerSize', 15, 'LineWidth', 2);
plot(PROBE1(1), PROBE1(2), 'mo', 'MarkerSize', 10, 'LineWidth', 2);
plot(PROBE2(1), PROBE2(2), 'co', 'MarkerSize', 10, 'LineWidth', 2);
hold off;
axis([1 NX 1 NY]);
axis image;
title('Velocity vector field');
xlabel('x'); ylabel('y');

sgtitle(sprintf('Thermal Obstacle Array Demo — Single Source (T1=%dC, T2=%dC)', T1_demo, T2_demo));

%% ============================================================
% 6b. Heatmap: probe velocity ratio vs (T1, T2)
%% ============================================================
figure('Position', [100 650 1200 400]);

subplot(1, 3, 1);
imagesc(T2_vals, T1_vals, v1_map);
axis image; colormap(gca, parula); colorbar;
title('Probe 1 velocity (upper)');
xlabel('T2 [degC]'); ylabel('T1 [degC]');
set(gca, 'YDir', 'normal');

subplot(1, 3, 2);
imagesc(T2_vals, T1_vals, v2_map);
axis image; colormap(gca, parula); colorbar;
title('Probe 2 velocity (lower)');
xlabel('T2 [degC]'); ylabel('T1 [degC]');
set(gca, 'YDir', 'normal');

subplot(1, 3, 3);
imagesc(T2_vals, T1_vals, log10(ratio_map));
axis image; colormap(gca, jet); colorbar;
title('log_{10}(v_1 / v_2) — routing map');
xlabel('T2 [degC]'); ylabel('T1 [degC]');
set(gca, 'YDir', 'normal');
hold on;
% Mark cells with strongest routing to each probe
[~, max_idx] = max(ratio_map(:));
[max_i, max_j] = ind2sub([N1, N2], max_idx);
plot(T2_vals(max_j), T1_vals(max_i), 'w*', 'MarkerSize', 15, 'LineWidth', 2);
[~, min_idx] = min(ratio_map(:));
[min_i, min_j] = ind2sub([N1, N2], min_idx);
plot(T2_vals(min_j), T1_vals(min_i), 'ko', 'MarkerSize', 10, 'LineWidth', 2);
hold off;

sgtitle('Thermal Routing Sweep — Single Source');

%% ============================================================
% 7. Summary statistics
%% ============================================================
max_ratio = max(ratio_map(:));
min_ratio = min(ratio_map(:));
contrast = (max_ratio - min_ratio) / (max_ratio + min_ratio);

fprintf('\n========== RESULTS ==========\n');
fprintf('Max v1/v2 ratio: %.3f  (T1=%d, T2=%d)\n', max_ratio, T1_vals(max_i), T2_vals(max_j));
fprintf('Min v1/v2 ratio: %.3f  (T1=%d, T2=%d)\n', min_ratio, T1_vals(min_i), T2_vals(min_j));
fprintf('Contrast metric: %.3f\n', contrast);
fprintf('=============================\n');
