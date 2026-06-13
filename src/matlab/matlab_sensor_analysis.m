%% matlab_sensor_analysis.m
%% 风电机组新型智能消防系统 —— 小波去噪与多传感器信号可视化
%  功能：生成四类火灾工况的传感器数据，对比小波去噪与滑动平均效果，
%        并绘制多通道时序图，为课程设计报告提供高质量插图。
%  需要工具箱：Wavelet Toolbox

clear; close all; clc;

%% ── 1. 参数设置 ──
fs = 1;                     % 采样频率 (Hz)
T  = 60;                    % 信号时长 (s)
t  = (0:T-1)';              % 时间轴
rng(42);                    % 固定随机种子

% 四类工况
scenarios = {'正常工况', '部件过热', '阴燃早期', '明火发展'};

% 传感器通道
channels = {'温度 (℃)', '烟雾 (ppm)', '气体 (%LEL)', '红外热点 (℃)', '电流 (A)'};

%% ── 2. 生成仿真数据 ──
n_scenarios = length(scenarios);
n_channels  = length(channels);
data = zeros(T, n_channels, n_scenarios);

for s = 1:n_scenarios
    % 基准噪声
    noise_temp  = 0.15 * randn(T, 1);
    noise_smoke = 0.03 * randn(T, 1);
    noise_gas   = 0.02 * randn(T, 1);
    noise_ir    = 0.25 * randn(T, 1);
    noise_curr  = 0.1  * randn(T, 1);

    switch s
        case 1  % 正常工况
            base_temp  = 25 + 0.02 * t;
            base_smoke = 0.05 + 0.001 * t;
            base_gas   = 0.02;
            base_ir    = 26 + 0.02 * t;
            base_curr  = 15 + 0.3 * sin(0.05 * t);

        case 2  % 部件过热
            base_temp  = 25 + 0.5 * t;
            base_smoke = 0.1  + 0.005 * t;
            base_gas   = 0.03 + 0.002 * t;
            base_ir    = 26 + 0.8 * t;
            base_curr  = 15 + 0.04 * t + 2 * sin(0.08 * t);

        case 3  % 阴燃早期
            base_temp  = 25 + 0.1  * t;
            base_smoke = 0.3  + 0.08 * t;
            base_gas   = 0.1  + 0.04 * t;
            base_ir    = 26 + 0.15 * t;
            base_curr  = 15 + 0.02 * t + 1.5 * sin(0.06 * t);

        case 4  % 明火发展
            base_temp  = 25 + 1.5  * t       + 3 * sin(0.2 * t);
            base_smoke = 0.5  + 0.15 * t     + 0.4 * sin(0.12 * t);
            base_gas   = 0.2  + 0.08 * t     + 0.15 * sin(0.1 * t);
            base_ir    = 26 + 3.0  * t       + 6 * sin(0.15 * t);
            base_curr  = 15 + 0.15 * t       + 5 * sin(0.1  * t);
    end

    data(:, 1, s) = base_temp  + noise_temp;
    data(:, 2, s) = base_smoke + noise_smoke;
    data(:, 3, s) = base_gas   + noise_gas;
    data(:, 4, s) = base_ir    + noise_ir;
    data(:, 5, s) = base_curr  + noise_curr;
end

%% ── 3. 小波去噪 ──

%%  ── 3. 小波去噪对比 ──
%  选取"明火发展"的温度通道做去噪对比演示
raw_signal = data(:, 1, 4);

% 滑动平均 (3点)
smooth_signal = movmean(raw_signal, 3);

% 小波去噪 (db4, 3层分解，软阈值)
wname = 'db4';
level = 3;
[thr, sorh, keepapp] = ddencmp('den', 'wv', raw_signal);
wavelet_denoised = wdenoise(raw_signal, level, ...
    'Wavelet', wname, ...
    'DenoisingMethod', 'Bayes', ...
    'ThresholdRule', 'Soft');

% 小波系数分解
[c, l] = wavedec(raw_signal, level, wname);

%% ── 4. 绘制去噪对比图 ──
figure('Position', [100, 100, 1200, 600], 'Color', 'white');

subplot(2,3,1);
plot(t, raw_signal, '-', 'Color', [0.7 0.7 0.7], 'LineWidth', 1.2); hold on;
plot(t, smooth_signal, 'b-', 'LineWidth', 1.8);
plot(t, wavelet_denoised, 'r-', 'LineWidth', 1.8);
xlabel('时间 (s)', 'FontSize', 10);
ylabel('温度 (℃)', 'FontSize', 10);
title('(a) 去噪对比：明火发展-温度通道', 'FontSize', 11, 'FontWeight', 'bold');
legend({'原始信号', '3点滑动平均', '小波去噪 (db4-Bayes)'}, ...
    'FontSize', 8, 'Location', 'northwest');
grid on; box off;

% 小波分解系数展示
for i = 1:3
    subplot(2,3,i+1);
    coeff = detcoef(c, l, i);
    plot(coeff, 'LineWidth', 1);
    xlabel('系数索引', 'FontSize', 9);
    ylabel(['d_', num2str(i)], 'FontSize', 9);
    title(['(b) 细节系数: 第', num2str(i), '层'], 'FontSize', 10, 'FontWeight', 'bold');
    grid on; box off;
end

subplot(2,3,5);
approx = appcoef(c, l, wname, level);
plot(approx, 'g-', 'LineWidth', 1.5);
xlabel('系数索引', 'FontSize', 9);
ylabel('A_3', 'FontSize', 9);
title('(c) 逼近系数 A_3', 'FontSize', 10, 'FontWeight', 'bold');
grid on; box off;

% 频谱对比
subplot(2,3,6);
[P_raw, f_raw] = pwelch(raw_signal, [], [], [], fs);
[P_wave, f_wave] = pwelch(wavelet_denoised, [], [], [], fs);
plot(f_raw, 10*log10(P_raw), 'Color', [0.6 0.6 0.6], 'LineWidth', 1.2); hold on;
plot(f_wave, 10*log10(P_wave), 'r-', 'LineWidth', 1.8);
xlabel('频率 (Hz)', 'FontSize', 9);
ylabel('功率谱密度 (dB/Hz)', 'FontSize', 9);
title('(d) 功率谱密度对比', 'FontSize', 10, 'FontWeight', 'bold');
legend({'原始信号', '小波去噪后'}, 'FontSize', 8);
grid on; box off;

sgtitle('图A-1  小波去噪与滑动平均滤波效果对比', 'FontSize', 14, 'FontWeight', 'bold');

% 保存图片
saveas(gcf, '小波去噪对比图.png');

%% ── 5. 四类工况多传感器时序面板图 ──
figure('Position', [50, 50, 1600, 900], 'Color', 'white');

for s = 1:n_scenarios
    for ch = 1:n_channels
        subplot(n_scenarios, n_channels, (s-1)*n_channels + ch);
        
        % 去噪后的数据
        denoised = wdenoise(data(:, ch, s), level, ...
            'Wavelet', wname, 'DenoisingMethod', 'Bayes', 'ThresholdRule', 'Soft');

        plot(t, data(:, ch, s), '-', 'Color', [0.75 0.75 0.75], 'LineWidth', 1); hold on;
        plot(t, denoised, '-', 'LineWidth', 1.6, 'Color', [0.1 0.35 0.6]);

        xlabel('时间 (s)', 'FontSize', 8);
        if ch == 1
            ylabel([scenarios{s}], 'FontSize', 10, 'FontWeight', 'bold');
        end
        if s == 1
            title(channels{ch}, 'FontSize', 10, 'FontWeight', 'bold');
        end
        grid on; box off;
        
        if s == n_scenarios && ch == n_channels
            legend({'原始', '小波去噪'}, 'FontSize', 7, 'Location', 'best');
        end
    end
end

sgtitle('图A-2  四类工况下多传感器时序信号（60 s）', 'FontSize', 15, 'FontWeight', 'bold');
saveas(gcf, '多传感器时序图.png');

%% ── 6. 输出汇总 ──
fprintf('小波去噪分析完成。\n');
fprintf('  去噪方法：db4 小波，3层分解，Bayes阈值，软阈值\n');
fprintf('  四类工况 × 5个传感器通道 = 20 条信号均已去噪并可视化\n');
fprintf('  图片已保存：小波去噪对比图.png, 多传感器时序图.png\n');

disp(' ');
disp('===== 脚本 matlab_sensor_analysis.m 运行完毕 =====');
