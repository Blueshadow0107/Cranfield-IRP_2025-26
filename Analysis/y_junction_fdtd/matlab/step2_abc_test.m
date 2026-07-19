%% Step 2: Absorbing Boundary Conditions (sponge layer)
% Add a damping layer near domain boundaries to absorb outgoing waves.
% Test by injecting a pulse and checking reflection amplitude.

clear; clc; close all;

%% Parameters (same as Step 1)
NX = 100;
NY = 100;
DX = 1e-4;
C = 1500.0;
CFL = 0.5;
DT = CFL * DX / (C * sqrt(2));
coef = (C * DT / DX)^2;

%% Sponge layer parameters
SPONGE_WIDTH = 15;   % Cells wide
SPONGE_DAMP = 0.3;   % Damping coefficient per step

sponge = build_sponge_mask(NX, NY, SPONGE_WIDTH, SPONGE_DAMP);

%% Source (Gaussian pulse)
SRC_X = round(NX / 2);
SRC_Y = round(NY / 2);
PULSE_WIDTH = 20 * DT;
PULSE_DELAY = 50 * DT;
SRC_AMP = 1.0;

%% Run with and without sponge
NT = 600;

% Probe near boundary (where reflections would arrive)
PROBE_X = 5;
PROBE_Y = round(NY / 2);

fprintf('Running without sponge...\n');
[trace_no_sponge, max_no] = run_simulation(false, NT, NX, NY, SRC_X, SRC_Y, PROBE_X, PROBE_Y, SRC_AMP, PULSE_DELAY, PULSE_WIDTH, DT, coef, sponge);
fprintf('Running with sponge...\n');
[trace_sponge, max_sp] = run_simulation(true, NT, NX, NY, SRC_X, SRC_Y, PROBE_X, PROBE_Y, SRC_AMP, PULSE_DELAY, PULSE_WIDTH, DT, coef, sponge);

%% Analysis: quantify reflection
t = (0:NT-1) * DT * 1e6;  % µs

% Direct arrival time from source to probe
dist = sqrt((SRC_X - PROBE_X)^2 + (SRC_Y - PROBE_Y)^2) * DX;
arrival = dist / C * 1e6;       % µs
% Reflection arrival time (source → far wall → probe)
refl_dist = ((NX - PROBE_X) + (NX - SRC_X)) * DX;
refl_arrival = refl_dist / C * 1e6 + arrival;

fprintf('\nDirect arrival at probe: ~%.2f µs\n', arrival);
fprintf('Reflected arrival at probe: ~%.2f µs\n', refl_arrival);

% Find peak before and after reflection
before_mask = t < refl_arrival - 1;
after_mask = t > refl_arrival + 1;

peak_before_no = max(abs(trace_no_sponge(before_mask)));
peak_after_no = max(abs(trace_no_sponge(after_mask)));
peak_before_sp = max(abs(trace_sponge(before_mask)));
peak_after_sp = max(abs(trace_sponge(after_mask)));

refl_ratio_no = peak_after_no / peak_before_no;
refl_ratio_sp = peak_after_sp / peak_before_sp;

fprintf('\nWithout sponge: reflected peak / direct peak = %.3f\n', refl_ratio_no);
fprintf('With sponge:    reflected peak / direct peak = %.3f\n', refl_ratio_sp);

%% Plot
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

figure('Position', [100 100 1200 800], 'Color', 'w');

% Probe traces
subplot(2, 2, 1);
plot(t, trace_no_sponge, 'b-', 'LineWidth', 1.0, 'DisplayName', 'No sponge');
hold on;
plot(t, trace_sponge, 'r-', 'LineWidth', 1.0, 'DisplayName', 'With sponge');
xline(refl_arrival, 'k--', 'Alpha', 0.5, 'Label', 'Expected reflection');
xlabel('t [µs]');
ylabel('p [Pa]');
title(sprintf('Probe at boundary (x=%.1f mm)', PROBE_X*DX*1e3));
legend('Location', 'best');
grid on;

% Zoomed probe traces
subplot(2, 2, 2);
plot(t, trace_no_sponge, 'b-', 'LineWidth', 1.0, 'DisplayName', 'No sponge');
hold on;
plot(t, trace_sponge, 'r-', 'LineWidth', 1.0, 'DisplayName', 'With sponge');
xline(refl_arrival, 'k--', 'Alpha', 0.5);
xlim([refl_arrival - 2, t(end)]);
xlabel('t [µs]');
ylabel('p [Pa]');
title('Zoom: reflection region');
legend('Location', 'best');
grid on;

% Sponge mask
subplot(2, 2, 3);
imagesc([0 NX*DX*1e3], [0 NY*DX*1e3], sponge.');
axis image;
set(gca, 'YDir', 'normal');
colormap(gca, hot);
colorbar;
title('Sponge damping mask');
xlabel('x [mm]');
ylabel('y [mm]');

% Max amplitude over time
subplot(2, 2, 4);
plot(t, max_no, 'b-', 'LineWidth', 1.0, 'DisplayName', 'No sponge');
hold on;
plot(t, max_sp, 'r-', 'LineWidth', 1.0, 'DisplayName', 'With sponge');
xlabel('t [µs]');
ylabel('max |p| [Pa]');
title('Global max amplitude');
legend('Location', 'best');
grid on;

sgtitle('Step 2: Absorbing Boundary Conditions', 'FontSize', 14);
outPath = fullfile(figDir, 'step2_abc_test.png');
print(gcf, outPath, '-dpng', '-r150');
fprintf('\nSaved figure: %s\n', outPath);

fprintf('\n--- Step 2 Complete ---\n');
fprintf('Checks:\n');
fprintf('  - Reflection without sponge: %.3f  (should be significant)\n', refl_ratio_no);
fprintf('  - Reflection with sponge:    %.3f  (should be < 0.05)\n', refl_ratio_sp);
if refl_ratio_sp < 0.05
    fprintf('  - Sponge effective: ✓\n');
else
    fprintf('  - Sponge may need tuning (increase SPONGE_DAMP or SPONGE_WIDTH)\n');
end

%% Local functions
function mask = build_sponge_mask(nx, ny, width, damp)
    mask = zeros(nx, ny);
    for i = 1:nx
        for j = 1:ny
            % Distance to nearest boundary (0-based distance)
            d = min([i-1, j-1, nx-i, ny-j]);
            if d < width
                mask(i, j) = damp * (1 - d / width)^2;
            end
        end
    end
end

function [probe_trace, max_p] = run_simulation(use_sponge, NT, NX, NY, SRC_X, SRC_Y, PROBE_X, PROBE_Y, SRC_AMP, PULSE_DELAY, PULSE_WIDTH, DT, coef, sponge)
    p_prev = zeros(NX, NY);
    p_curr = zeros(NX, NY);
    p_next = zeros(NX, NY);
    probe_trace = zeros(1, NT);
    max_p = zeros(1, NT);
    
    for n = 0:NT-1
        t = n * DT;
        p_curr(SRC_X, SRC_Y) = p_curr(SRC_X, SRC_Y) + SRC_AMP * exp(-((t - PULSE_DELAY) / PULSE_WIDTH)^2);
        
        % Simple 5-point stencil (equivalent to np.roll)
        laplacian = circshift(p_curr, [1, 0]) + circshift(p_curr, [-1, 0]) + ...
                    circshift(p_curr, [0, 1]) + circshift(p_curr, [0, -1]) - 4 * p_curr;
        
        p_next = 2 * p_curr - p_prev + coef * laplacian;
        
        if use_sponge
            % Apply sponge damping to velocity-like term
            p_next = p_next - sponge .* (p_next - p_prev);
        end
        
        % Swap
        temp = p_prev;
        p_prev = p_curr;
        p_curr = p_next;
        p_next = temp;
        
        probe_trace(n+1) = p_curr(PROBE_X, PROBE_Y);
        max_p(n+1) = max(abs(p_curr(:)));
    end
end
