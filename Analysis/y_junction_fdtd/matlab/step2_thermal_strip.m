%% Step 2: Phase Delay from a Heated Strip
% Run A (T=0) vs Run B (T=50C in strip). Measure phase delay at probe.

clc; clear; close all;

% Parameters
DX = 5e-5; C0 = 1500.0; RHO0 = 998.0;
CFL = 0.5; DT = CFL * DX / (C0 * sqrt(2));
NX = 200; NY = 200;

SRC_X = 30; SRC_Y = 100;
FREQ = 1.0e6; BURST_CYCLES = 10;
BURST_DURATION = BURST_CYCLES / FREQ;
BURST_DELAY = 2.0e-6; SRC_AMP = 10.0;

PROBE_X = 180; PROBE_Y = 100;

STRIP_X1 = 80; STRIP_X2 = 120;
STRIP_Y1 = 60; STRIP_Y2 = 140;

ALPHA_C = 2.4e-3; T_STRIP = 50.0;

TRAVEL_TIME = (PROBE_X - SRC_X) * DX / C0;
NT = round((BURST_DELAY + BURST_DURATION + 2*TRAVEL_TIME) / DT);

fprintf('Grid: %d x %d, DX = %.2e, DT = %.2e\n', NX, NY, DX, DT);
fprintf('Timesteps: %d, Sim time: %.2f us\n', NT, NT*DT*1e6);
fprintf('Strip: x=%d-%d, y=%d-%d, T=%.1f C\n', STRIP_X1, STRIP_X2, STRIP_Y1, STRIP_Y2, T_STRIP);

% Thermal fields
c_field_A = C0 * ones(NX, NY);
T_field_B = zeros(NX, NY);
T_field_B(STRIP_X1:STRIP_X2, STRIP_Y1:STRIP_Y2) = T_STRIP;
c_field_B = C0 * ones(NX, NY);
c_field_B(STRIP_X1:STRIP_X2, STRIP_Y1:STRIP_Y2) = C0 * (1 - ALPHA_C * T_STRIP);

press_coef_A = RHO0 * (c_field_A.^2);
press_coef_B = RHO0 * (c_field_B.^2);

% Helpers
function val = burst_source(t, delay, duration, freq, amp)
    if t < delay || t > delay + duration, val = 0; return; end
    tau = (t - delay) / duration;
    envelope = 0.5 * (1 - cos(2*pi*tau));
    val = amp * envelope * sin(2*pi*freq*(t - delay));
end

function a = analytic_signal(x)
    N = length(x); X = fft(x);
    h = zeros(N, 1); h(1) = 1;
    if mod(N, 2) == 0, h(2:N/2) = 2; h(N/2 + 1) = 1;
    else, h(2:(N+1)/2) = 2; end
    a = ifft(X .* h);
end

function [xc, lags] = cross_correlate(x, y, maxlag)
    N = length(x); M = length(y);
    lags = -maxlag:maxlag; xc = zeros(length(lags), 1);
    for i = 1:length(lags)
        lag = lags(i);
        if lag < 0
            len = min(N + lag, M);
            if len > 0, xc(i) = sum(x(1:len) .* conj(y(1-lag:1-lag+len-1))); end
        else
            len = min(N - lag, M);
            if len > 0, xc(i) = sum(x(lag+1:lag+len) .* conj(y(1:len))); end
        end
    end
end

function phi = wrap_to_pi(phi)
    phi = mod(phi + pi, 2*pi) - pi;
end

function [probe_signal, p_final, vx_final, vy_final] = run_fdtd(press_coef, NT, NX, NY, DX, DT, RHO0, SRC_X, SRC_Y, PROBE_X, PROBE_Y, BURST_DELAY, BURST_DURATION, FREQ, SRC_AMP)
    p = zeros(NX, NY); vx = zeros(NX, NY); vy = zeros(NX, NY);
    probe_signal = zeros(NT, 1);
    for n = 1:NT
        t = (n-1) * DT;
        
        p_pad = zeros(NX+2, NY+2);
        p_pad(2:end-1, 2:end-1) = p;
        p_pad(1, :) = p_pad(2, :); p_pad(end, :) = p_pad(end-1, :);
        p_pad(:, 1) = p_pad(:, 2); p_pad(:, end) = p_pad(:, end-1);
        
        dP_dx = (p_pad(3:end, 2:end-1) - p_pad(1:end-2, 2:end-1)) / (2*DX);
        dP_dy = (p_pad(2:end-1, 3:end) - p_pad(2:end-1, 1:end-2)) / (2*DX);
        vx = vx - (DT/RHO0) * dP_dx;
        vy = vy - (DT/RHO0) * dP_dy;
        vx(1, :) = 0; vx(end, :) = 0; vy(:, 1) = 0; vy(:, end) = 0;
        
        vx_pad = zeros(NX+2, NY+2); vy_pad = zeros(NX+2, NY+2);
        vx_pad(2:end-1, 2:end-1) = vx; vy_pad(2:end-1, 2:end-1) = vy;
        vx_pad(1, :) = -vx_pad(2, :); vx_pad(end, :) = -vx_pad(end-1, :);
        vx_pad(:, 1) = vx_pad(:, 2); vx_pad(:, end) = vx_pad(:, end-1);
        vy_pad(:, 1) = -vy_pad(:, 2); vy_pad(:, end) = -vy_pad(:, end-1);
        vy_pad(1, :) = vy_pad(2, :); vy_pad(end, :) = vy_pad(end-1, :);
        
        div_v = (vx_pad(3:end, 2:end-1) - vx_pad(1:end-2, 2:end-1)) / (2*DX) ...
              + (vy_pad(2:end-1, 3:end) - vy_pad(2:end-1, 1:end-2)) / (2*DX);
        p = p - DT * press_coef .* div_v;
        
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
        
        probe_signal(n) = p(PROBE_X, PROBE_Y);
    end
    p_final = p; vx_final = vx; vy_final = vy;
end

% Run both simulations
fprintf('\n--- Running A (no heating) ---\n');
[probe_A, p_A, vx_A, vy_A] = run_fdtd(press_coef_A, NT, NX, NY, DX, DT, RHO0, SRC_X, SRC_Y, PROBE_X, PROBE_Y, BURST_DELAY, BURST_DURATION, FREQ, SRC_AMP);
fprintf('--- Running B (heated strip) ---\n');
[probe_B, p_B, vx_B, vy_B] = run_fdtd(press_coef_B, NT, NX, NY, DX, DT, RHO0, SRC_X, SRC_Y, PROBE_X, PROBE_Y, BURST_DELAY, BURST_DURATION, FREQ, SRC_AMP);

% Phase delay analysis
t_us = (0:NT-1) * DT * 1e6;
window_start = max(1, round(BURST_DELAY / DT));
window_end = min(NT, round((BURST_DELAY + BURST_DURATION + 2*TRAVEL_TIME) / DT));
sig_A = probe_A(window_start:window_end);
sig_B = probe_B(window_start:window_end);

L_strip = (STRIP_X2 - STRIP_X1 + 1) * DX;
c_hot = C0 * (1 - ALPHA_C * T_STRIP); c_cold = C0;
dt_theory = L_strip * (1/c_hot - 1/c_cold);
phase_theory = 360 * FREQ * dt_theory;

% Method 1: Envelope peak
env_A = abs(analytic_signal(sig_A)); env_B = abs(analytic_signal(sig_B));
[~, peak_A] = max(env_A); [~, peak_B] = max(env_B);
dt_env = (peak_B - peak_A) * DT; phase_env = 360 * FREQ * dt_env;

% Method 2: Restricted cross-correlation
max_lag_us = 0.5e-6; max_lag_samples = round(max_lag_us / DT);
[xc, lags] = cross_correlate(sig_B, sig_A, max_lag_samples);
[~, max_idx] = max(abs(xc));
dt_xcorr = lags(max_idx) * DT; phase_xcorr = 360 * FREQ * dt_xcorr;

% Method 3: FFT phase
Nfft = 2^nextpow2(length(sig_A) * 4);
fft_A = fft(sig_A, Nfft); fft_B = fft(sig_B, Nfft);
freq_ax = (0:Nfft-1) / (Nfft * DT);
[~, f_idx] = min(abs(freq_ax - FREQ));
phase_diff = wrap_to_pi(angle(fft_B(f_idx)) - angle(fft_A(f_idx)));
dt_fft = phase_diff / (2*pi*FREQ); phase_fft = phase_diff * 180/pi;

fprintf('\n========== PHASE DELAY RESULTS ==========\n');
fprintf('Strip: %.3f mm, c_hot = %.1f, c_cold = %.1f\n', L_strip*1e3, c_hot, c_cold);
fprintf('Theory:  dt = %.4f us, phase = %.2f deg\n', dt_theory*1e6, phase_theory);
fprintf('Envelope: dt = %.4f us, phase = %.2f deg\n', dt_env*1e6, phase_env);
fprintf('Xcorr:   dt = %.4f us, phase = %.2f deg\n', dt_xcorr*1e6, phase_xcorr);
fprintf('FFT:     dt = %.4f us, phase = %.2f deg\n', dt_fft*1e6, phase_fft);
fprintf('=========================================\n');

% Plots
figure('Name', 'Sound Speed', 'Position', [100 100 600 400]);
imagesc(c_field_B'); axis image; colormap(gca, jet); colorbar;
title('Sound Speed c(x,y) for Run B'); xlabel('x [cells]'); ylabel('y [cells]');
hold on; plot(SRC_X, SRC_Y, 'w*', 'MarkerSize', 15); plot(PROBE_X, PROBE_Y, 'wo', 'MarkerSize', 12);
rectangle('Position', [STRIP_X1, STRIP_Y1, STRIP_X2-STRIP_X1, STRIP_Y2-STRIP_Y1], 'EdgeColor', 'w', 'LineWidth', 2, 'LineStyle', '--');
hold off;

figure('Name', 'Probe Traces', 'Position', [100 100 900 400]);
plot(t_us, probe_A, 'b-', 'LineWidth', 1.5); hold on;
plot(t_us, probe_B, 'r-', 'LineWidth', 1.5);
xlabel('Time [us]'); ylabel('Pressure [Pa]'); title('Probe: No Heat (blue) vs Heated (red)');
legend('Run A: T=0', 'Run B: T=50C', 'Location', 'best'); grid on; xlim([0, t_us(end)]);

zoom_start = max(1, round((BURST_DELAY + 0.8*TRAVEL_TIME) / DT));
zoom_end = min(NT, round((BURST_DELAY + BURST_DURATION + 1.5*TRAVEL_TIME) / DT));
t_zoom = t_us(zoom_start:zoom_end);

figure('Name', 'Zoomed', 'Position', [100 100 900 500]);
subplot(2, 1, 1);
plot(t_zoom, probe_A(zoom_start:zoom_end), 'b-', 'LineWidth', 1.5); hold on;
plot(t_zoom, probe_B(zoom_start:zoom_end), 'r-', 'LineWidth', 1.5);
plot(t_zoom, env_A(zoom_start-window_start+1:zoom_end-window_start+1), 'b--', 'LineWidth', 2);
plot(t_zoom, env_B(zoom_start-window_start+1:zoom_end-window_start+1), 'r--', 'LineWidth', 2);
xlabel('Time [us]'); ylabel('Pressure [Pa]');
title(sprintf('Signals + Envelopes (env: %.3f us, theory: %.3f us)', dt_env*1e6, dt_theory*1e6));
legend('Run A', 'Run B', 'Env A', 'Env B', 'Location', 'best'); grid on;

subplot(2, 1, 2);
plot(t_zoom, probe_A(zoom_start:zoom_end), 'b-', 'LineWidth', 1.5); hold on;
plot(t_zoom, probe_B(zoom_start:zoom_end), 'r-', 'LineWidth', 1.5);
xlabel('Time [us]'); ylabel('Pressure [Pa]');
title(sprintf('Zoomed: FFT = %.1f deg, xcorr = %.1f deg, theory = %.1f deg', phase_fft, phase_xcorr, phase_theory));
legend('Run A: T=0', 'Run B: T=50C', 'Location', 'best'); grid on;

figure('Name', 'Xcorr Diagnostic', 'Position', [100 100 800 300]);
lags_us = lags * DT * 1e6;
plot(lags_us, abs(xc), 'k-', 'LineWidth', 1.5); hold on;
plot(lags_us(max_idx), abs(xc(max_idx)), 'ro', 'MarkerSize', 12, 'LineWidth', 2);
plot([0 0], ylim, 'g--', 'LineWidth', 1); plot(dt_theory*1e6 * [1 1], ylim, 'b--', 'LineWidth', 1);
xlabel('Lag [us]'); ylabel('|Cross-correlation|');
title(sprintf('Xcorr peak: %.3f us (theory: %.3f us)', dt_xcorr*1e6, dt_theory*1e6));
legend('|xcorr|', 'Peak', 'Zero lag', 'Theory', 'Location', 'best'); grid on;

figure('Name', 'Final Fields', 'Position', [100 100 1200 400]);
subplot(1, 2, 1); imagesc(p_A'); axis image; colormap(gca, jet); colorbar;
title('Run A: No Heating'); xlabel('x'); ylabel('y');
hold on; plot(SRC_X, SRC_Y, 'w*', 'MarkerSize', 12); plot(PROBE_X, PROBE_Y, 'wo', 'MarkerSize', 10);
rectangle('Position', [STRIP_X1, STRIP_Y1, STRIP_X2-STRIP_X1, STRIP_Y2-STRIP_Y1], 'EdgeColor', 'w', 'LineWidth', 1.5, 'LineStyle', '--');
hold off; caxis([-1 1] * SRC_AMP);

subplot(1, 2, 2); imagesc(p_B'); axis image; colormap(gca, jet); colorbar;
title('Run B: Heated Strip'); xlabel('x'); ylabel('y');
hold on; plot(SRC_X, SRC_Y, 'w*', 'MarkerSize', 12); plot(PROBE_X, PROBE_Y, 'wo', 'MarkerSize', 10);
rectangle('Position', [STRIP_X1, STRIP_Y1, STRIP_X2-STRIP_X1, STRIP_Y2-STRIP_Y1], 'EdgeColor', 'w', 'LineWidth', 1.5, 'LineStyle', '--');
hold off; caxis([-1 1] * SRC_AMP);
sgtitle('Pressure Fields at Final Timestep');
