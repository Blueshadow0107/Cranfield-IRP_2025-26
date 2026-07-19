%% Step 1: Velocity-Pressure FDTD in Empty Domain
% Validates: |v|/|p| = 1/(rho*c), 1/sqrt(r) decay in 2D.
% Physics: dp/dt = -rho0*c^2*div(v), dv/dt = -(1/rho0)*grad(p)

clc; clear; close all;

% Parameters
DX = 5e-5;                    % Grid spacing [m]
C0 = 1500.0;                  % Sound speed [m/s]
RHO0 = 998.0;                 % Density [kg/m^3]
CFL = 0.5;
DT = CFL * DX / (C0 * sqrt(2));

NX = 200; NY = 200;
SRC_X = round(NX/2); SRC_Y = round(NY/2);
FREQ = 1.0e6;
BURST_CYCLES = 10;
BURST_DURATION = BURST_CYCLES / FREQ;
BURST_DELAY = 2.0e-6;
SRC_AMP = 10.0;

COURANT_TIME = NX * DX / C0;
NT = round((BURST_DELAY + BURST_DURATION + 1.5*COURANT_TIME) / DT);

fprintf('Grid: %d x %d, DX = %.2e, DT = %.2e\n', NX, NY, DX, DT);
fprintf('Timesteps: %d, Sim time: %.2f us\n', NT, NT*DT*1e6);

% Fields (uniform c for Step 1)
press_coef = RHO0 * C0^2;
p = zeros(NX, NY);
vx = zeros(NX, NY); vy = zeros(NX, NY);

% Probes along +x from source
probe_radii = [10, 20, 30, 50, 70] * DX;
n_probes = length(probe_radii);
probe_idx = zeros(n_probes, 2);
for k = 1:n_probes
    r_cells = round(probe_radii(k) / DX);
    probe_idx(k, :) = [SRC_X + r_cells, SRC_Y];
end
p_hist = zeros(NT, n_probes);
v_hist = zeros(NT, n_probes);

% Burst source
function val = burst_source(t, delay, duration, freq, amp)
    if t < delay || t > delay + duration, val = 0; return; end
    tau = (t - delay) / duration;
    envelope = 0.5 * (1 - cos(2*pi*tau));
    val = amp * envelope * sin(2*pi*freq*(t - delay));
end

% Snapshots
n_snapshots = 9;
snap_times = round(linspace(1, NT, n_snapshots));
snap_p = cell(n_snapshots, 1); snap_vmag = cell(n_snapshots, 1);
snap_labels = cell(n_snapshots, 1); snap_count = 0;

% Main loop
for n = 1:NT
    t = (n-1) * DT;
    
    % Velocity update
    p_pad = zeros(NX+2, NY+2);
    p_pad(2:end-1, 2:end-1) = p;
    p_pad(1, :) = p_pad(2, :); p_pad(end, :) = p_pad(end-1, :);
    p_pad(:, 1) = p_pad(:, 2); p_pad(:, end) = p_pad(:, end-1);
    
    dP_dx = (p_pad(3:end, 2:end-1) - p_pad(1:end-2, 2:end-1)) / (2*DX);
    dP_dy = (p_pad(2:end-1, 3:end) - p_pad(2:end-1, 1:end-2)) / (2*DX);
    vx = vx - (DT/RHO0) * dP_dx;
    vy = vy - (DT/RHO0) * dP_dy;
    vx(1, :) = 0; vx(end, :) = 0; vy(:, 1) = 0; vy(:, end) = 0;
    
    % Pressure update
    vx_pad = zeros(NX+2, NY+2); vy_pad = zeros(NX+2, NY+2);
    vx_pad(2:end-1, 2:end-1) = vx; vy_pad(2:end-1, 2:end-1) = vy;
    vx_pad(1, :) = -vx_pad(2, :); vx_pad(end, :) = -vx_pad(end-1, :);
    vx_pad(:, 1) = vx_pad(:, 2); vx_pad(:, end) = vx_pad(:, end-1);
    vy_pad(:, 1) = -vy_pad(:, 2); vy_pad(:, end) = -vy_pad(:, end-1);
    vy_pad(1, :) = vy_pad(2, :); vy_pad(end, :) = vy_pad(end-1, :);
    
    div_v = (vx_pad(3:end, 2:end-1) - vx_pad(1:end-2, 2:end-1)) / (2*DX) ...
          + (vy_pad(2:end-1, 3:end) - vy_pad(2:end-1, 1:end-2)) / (2*DX);
    p = p - DT * press_coef * div_v;
    
    % Source injection (after pressure update)
    source_val = burst_source(t, BURST_DELAY, BURST_DURATION, FREQ, SRC_AMP);
    for di = -1:1
        for dj = -1:1
            ix = SRC_X + di; iy = SRC_Y + dj;
            if ix >= 1 && ix <= NX && iy >= 1 && iy <= NY
                weight = 1.0 / (1.0 + abs(di) + abs(dj));
                p(ix, iy) = p(ix, iy) + weight * source_val;
            end
        end
    end
    
    % Record probes
    for k = 1:n_probes
        p_hist(n, k) = p(probe_idx(k, 1), probe_idx(k, 2));
        v_hist(n, k) = sqrt(vx(probe_idx(k, 1), probe_idx(k, 2))^2 + vy(probe_idx(k, 1), probe_idx(k, 2))^2);
    end
    
    % Store snapshots
    if snap_count < n_snapshots && n >= snap_times(snap_count + 1)
        snap_count = snap_count + 1;
        snap_p{snap_count} = p;
        snap_vmag{snap_count} = sqrt(vx.^2 + vy.^2);
        snap_labels{snap_count} = sprintf('t = %.2f us', t*1e6);
    end
    
    if mod(n, 100) == 0
        fprintf('Step %d / %d (t = %.2f us)\n', n, NT, t*1e6);
    end
end

% Snapshot grids
figure('Name', 'Pressure Snapshots', 'Position', [100 100 1200 900]);
for k = 1:n_snapshots
    subplot(3, 3, k);
    imagesc(snap_p{k}'); axis image; colormap(gca, jet); colorbar;
    title(snap_labels{k}); xlabel('x'); ylabel('y');
    hold on; plot(SRC_X, SRC_Y, 'w*', 'MarkerSize', 12);
    for ip = 1:n_probes, plot(probe_idx(ip, 1), probe_idx(ip, 2), 'wo', 'MarkerSize', 8); end
    hold off; caxis([-1 1] * SRC_AMP);
end
sgtitle('Pressure Field Evolution');

figure('Name', 'Velocity Snapshots', 'Position', [100 100 1200 900]);
for k = 1:n_snapshots
    subplot(3, 3, k);
    imagesc(snap_vmag{k}'); axis image; colormap(gca, hot); colorbar;
    title(snap_labels{k}); xlabel('x'); ylabel('y');
    hold on; plot(SRC_X, SRC_Y, 'w*', 'MarkerSize', 12); hold off;
    caxis([0 1] * SRC_AMP / (RHO0 * C0));
end
sgtitle('Velocity Magnitude Evolution');

% Probe traces
figure('Name', 'Probe Time Traces', 'Position', [100 100 1000 700]);
t_us = (0:NT-1) * DT * 1e6;
subplot(2, 1, 1);
for k = 1:n_probes, plot(t_us, p_hist(:, k), 'LineWidth', 1.2); hold on; end
hold off; xlabel('Time [us]'); ylabel('Pressure [Pa]'); title('Pressure at Probes');
legend(arrayfun(@(r) sprintf('r=%.1f mm', r*1e3), probe_radii, 'UniformOutput', false), 'Location', 'best');
grid on; xlim([0, t_us(end)]);

subplot(2, 1, 2);
for k = 1:n_probes, plot(t_us, v_hist(:, k) * (RHO0 * C0), 'LineWidth', 1.2); hold on; end
hold off; xlabel('Time [us]'); ylabel('|v| * rho*c [Pa]'); title('Scaled Velocity at Probes');
legend(arrayfun(@(r) sprintf('r=%.1f mm', r*1e3), probe_radii, 'UniformOutput', false), 'Location', 'best');
grid on; xlim([0, t_us(end)]);

% Validation
burst_start_idx = round(BURST_DELAY / DT) + 1;
burst_end_idx = min(NT, round((BURST_DELAY + BURST_DURATION) / DT) + 1);
p_rms = zeros(n_probes, 1); v_rms = zeros(n_probes, 1);
for k = 1:n_probes
    p_rms(k) = sqrt(mean(p_hist(burst_start_idx:burst_end_idx, k).^2));
    v_rms(k) = sqrt(mean(v_hist(burst_start_idx:burst_end_idx, k).^2));
end
expected_ratio = 1 / (RHO0 * C0);

figure('Name', 'Validation', 'Position', [100 100 1000 400]);
subplot(1, 2, 1);
loglog(probe_radii * 1e3, p_rms, 'bo-', 'LineWidth', 2, 'MarkerSize', 10); hold on;
loglog(probe_radii * 1e3, v_rms * (RHO0 * C0), 'rs--', 'LineWidth', 2, 'MarkerSize', 10);
r_ref = linspace(probe_radii(1), probe_radii(end), 100);
loglog(r_ref * 1e3, p_rms(1) * sqrt(probe_radii(1) ./ r_ref), 'k:', 'LineWidth', 1.5);
hold off; xlabel('Radius [mm]'); ylabel('RMS amplitude'); title('2D Decay Validation');
legend('p', '|v|\rho c', '1/\surd r ref', 'Location', 'best'); grid on;

subplot(1, 2, 2);
actual_ratio = v_rms ./ p_rms;
bar(1:n_probes, actual_ratio, 'FaceColor', [0.2 0.6 0.8]); hold on;
plot([0.5, n_probes+0.5], [expected_ratio, expected_ratio], 'r--', 'LineWidth', 2);
hold off; xlabel('Probe index'); ylabel('|v| / |p|');
title(sprintf('Ratio (theory: %.4e)', expected_ratio));
set(gca, 'XTick', 1:n_probes, 'XTickLabel', arrayfun(@(r) sprintf('%.1f', r*1e3), probe_radii, 'UniformOutput', false));
grid on;

fprintf('\n========== VALIDATION RESULTS ==========\n');
fprintf('Expected |v|/|p| = %.6e\n', expected_ratio);
for k = 1:n_probes
    if p_rms(k) > 0
        fprintf('Probe %d (r=%.2f mm): |v|/|p| = %.6e, error = %.2f%%\n', ...
            k, probe_radii(k)*1e3, actual_ratio(k), abs(actual_ratio(k)-expected_ratio)/expected_ratio*100);
    else
        fprintf('Probe %d (r=%.2f mm): NO SIGNAL\n', k, probe_radii(k)*1e3);
    end
end
fprintf('========================================\n');
