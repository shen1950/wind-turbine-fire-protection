%% matlab_fuzzy_fis.m
%% 风电机组新型智能消防系统 —— 模糊推理系统(FIS)设计
%  功能：使用Fuzzy Logic Toolbox构建5输入-1输出的Mamdani模糊系统，
%        输出综合风险指数R(t)，并可视化隶属度函数与控制曲面。
%  需要工具箱：Fuzzy Logic Toolbox

clear; close all; clc;

%% ── 1. 创建Mamdani型模糊推理系统 ──
fis = mamfis('Name', 'WindTurbineFireRisk', ...
             'AndMethod', 'min', ...
             'OrMethod',  'max', ...
             'ImplicationMethod', 'min', ...
             'AggregationMethod', 'sum', ...
             'DefuzzificationMethod', 'centroid');

%% ── 2. 定义输入变量 ──

% 输入1：温度归一化值
fis = addInput(fis, [0 1], 'Name', 'Temperature');
fis = addMF(fis, 'Temperature', 'gaussmf', [0.12 0.0],  'Name', 'Low');
fis = addMF(fis, 'Temperature', 'gaussmf', [0.10 0.5],  'Name', 'Medium');
fis = addMF(fis, 'Temperature', 'gaussmf', [0.12 1.0],  'Name', 'High');

% 输入2：温升速率归一化值
fis = addInput(fis, [0 1], 'Name', 'TempRate');
fis = addMF(fis, 'TempRate', 'gaussmf', [0.12 0.0],  'Name', 'Slow');
fis = addMF(fis, 'TempRate', 'gaussmf', [0.10 0.5],  'Name', 'Moderate');
fis = addMF(fis, 'TempRate', 'gaussmf', [0.12 1.0],  'Name', 'Fast');

% 输入3：烟雾浓度归一化值
fis = addInput(fis, [0 1], 'Name', 'Smoke');
fis = addMF(fis, 'Smoke', 'gaussmf', [0.12 0.0],  'Name', 'Low');
fis = addMF(fis, 'Smoke', 'gaussmf', [0.10 0.5],  'Name', 'Medium');
fis = addMF(fis, 'Smoke', 'gaussmf', [0.12 1.0],  'Name', 'High');

% 输入4：气体浓度归一化值
fis = addInput(fis, [0 1], 'Name', 'Gas');
fis = addMF(fis, 'Gas', 'gaussmf', [0.12 0.0],  'Name', 'Low');
fis = addMF(fis, 'Gas', 'gaussmf', [0.10 0.5],  'Name', 'Medium');
fis = addMF(fis, 'Gas', 'gaussmf', [0.12 1.0],  'Name', 'High');

% 输入5：LSTM火灾概率
fis = addInput(fis, [0 1], 'Name', 'LSTM_Prob');
fis = addMF(fis, 'LSTM_Prob', 'gaussmf', [0.10 0.0],  'Name', 'Low');
fis = addMF(fis, 'LSTM_Prob', 'gaussmf', [0.12 0.5],  'Name', 'Medium');
fis = addMF(fis, 'LSTM_Prob', 'gaussmf', [0.10 1.0],  'Name', 'High');

%% ── 3. 定义输出变量 ──
fis = addOutput(fis, [0 1], 'Name', 'RiskIndex');
fis = addMF(fis, 'RiskIndex', 'trimf', [0.0 0.0 0.25], 'Name', 'Safe');
fis = addMF(fis, 'RiskIndex', 'trimf', [0.1 0.35 0.55], 'Name', 'Attention');
fis = addMF(fis, 'RiskIndex', 'trimf', [0.45 0.65 0.85], 'Name', 'Warning');
fis = addMF(fis, 'RiskIndex', 'trimf', [0.75 1.0 1.0], 'Name', 'Alarm');

%% ── 4. 定义模糊规则 ──
ruleList = [
    % Temperature    TempRate    Smoke    Gas    LSTM_Prob    RiskIndex    Weight    Conn
    % ───── 单变量轻度异常 → 关注 ─────
    1 1 1 1 1   1   1   1;  % 全低 → Safe
    2 1 1 1 1   2   1   1;  % 温度中 → Attention
    3 1 1 1 1   3   1   1;  % 温度高 → Warning
    1 2 1 1 1   2   1   1;  % 温升中 → Attention
    1 3 1 1 1   3   1   1;  % 温升快 → Warning
    1 1 2 1 1   2   1   1;  % 烟中 → Attention
    1 1 1 2 1   2   1   1;  % 气中 → Attention

    % ───── 多变量一致异常 → 预警/告警 ─────
    2 2 1 1 1   2   1   1;  % 温中+温升中 → Attention
    3 2 1 1 1   3   1   1;  % 温高+温升中 → Warning
    2 2 2 1 1   3   1   1;  % 温中+温升中+烟中 → Warning
    3 3 2 1 1   3   1   1;  % 温高+温升快+烟中 → Warning
    3 3 3 3 1   4   1   1;  % 温高+温升快+烟高+气高 → Alarm

    % ───── LSTM与传感器融合 ─────
    1 1 1 1 3   3   1   1;  % 全低+LSTM高 → Warning
    2 2 2 2 3   4   1   1;  % 多变量中+LSTM高 → Alarm
    3 3 3 3 3   4   1   1;  % 全高+LSTM高 → Alarm
    2 1 2 1 2   2   1   1;  % 温中+烟中+LSTM中 → Attention
    2 2 2 2 2   3   1   1;  % 多中+LSTM中 → Warning
];

fis = addRule(fis, ruleList);

fprintf('模糊推理系统构建完成。\n');
fprintf('  输入：5个（温度、温升速率、烟雾、气体、LSTM概率）\n');
fprintf('  输出：1个（综合风险指数）\n');
fprintf('  规则：%d 条\n', size(ruleList, 1));

%% ── 5. 可视化隶属度函数 ──

% 输入隶属度
figure('Position', [100, 100, 1300, 750], 'Color', 'white');
inputNames = {'Temperature', 'TempRate', 'Smoke', 'Gas', 'LSTM_Prob'};
inputLabels = {'(a) 温度归一化值', '(b) 温升速率归一化值', ...
               '(c) 烟雾浓度归一化值', '(d) 气体浓度归一化值', ...
               '(e) LSTM火灾概率'};

for i = 1:5
    subplot(2, 3, i);
    plotmf(fis, 'input', i);
    xlabel(inputLabels{i}, 'FontSize', 9);
    ylabel('隶属度', 'FontSize', 9);
    title(['输入', num2str(i), ': ', inputNames{i}], 'FontSize', 10, 'FontWeight', 'bold');
    grid on; box off;
    set(gca, 'FontSize', 8);
end

% 输出隶属度
subplot(2, 3, 6);
plotmf(fis, 'output', 1);
xlabel('(f) 综合风险指数 R(t)', 'FontSize', 9);
ylabel('隶属度', 'FontSize', 9);
title('输出: RiskIndex', 'FontSize', 10, 'FontWeight', 'bold');
grid on; box off;
set(gca, 'FontSize', 8);

sgtitle('图B-1  模糊推理系统 (Mamdani FIS) 隶属度函数', ...
    'FontSize', 14, 'FontWeight', 'bold');
saveas(gcf, '模糊隶属度函数.png');

%% ── 6. 绘制控制曲面（部分投影） ──
figure('Position', [50, 50, 1500, 500], 'Color', 'white');

% 曲面1: 温度 vs 烟雾 (其他为0.5)
subplot(1, 3, 1);
gensurf(fis, [1 3], 1);
xlabel('温度 (归一化)', 'FontSize', 9);
ylabel('烟雾 (归一化)', 'FontSize', 9);
zlabel('R(t)', 'FontSize', 9);
title('(a) 温度 vs 烟雾 → R(t)', 'FontSize', 10, 'FontWeight', 'bold');
view(45, 30); grid on;

% 曲面2: 温度 vs LSTM概率 (其他为0.5)
subplot(1, 3, 2);
gensurf(fis, [1 5], 1);
xlabel('温度 (归一化)', 'FontSize', 9);
ylabel('LSTM概率', 'FontSize', 9);
zlabel('R(t)', 'FontSize', 9);
title('(b) 温度 vs LSTM概率 → R(t)', 'FontSize', 10, 'FontWeight', 'bold');
view(45, 30); grid on;

% 曲面3: 温升速率 vs 烟雾 (其他为0.5)
subplot(1, 3, 3);
gensurf(fis, [2 3], 1);
xlabel('温升速率 (归一化)', 'FontSize', 9);
ylabel('烟雾 (归一化)', 'FontSize', 9);
zlabel('R(t)', 'FontSize', 9);
title('(c) 温升速率 vs 烟雾 → R(t)', 'FontSize', 10, 'FontWeight', 'bold');
view(45, 30); grid on;

sgtitle('图B-2  模糊推理系统控制曲面（部分输入投影）', ...
    'FontSize', 14, 'FontWeight', 'bold');
saveas(gcf, '模糊控制曲面.png');

%% ── 7. 测试样例 ──
fprintf('\n===== FIS 测试样例 =====\n');
testCases = {
    '正常运行',     [0.10, 0.05, 0.05, 0.02, 0.12];
    '轻微异常',     [0.40, 0.30, 0.20, 0.10, 0.25];
    '预警状态',     [0.60, 0.50, 0.55, 0.40, 0.60];
    '高危状态',     [0.85, 0.80, 0.90, 0.85, 0.92];
};

for i = 1:size(testCases, 1)
    inputs = testCases{i, 2};
    risk = evalfis(fis, inputs);
    fprintf('  %-10s  T=%.2f  dT=%.2f  Sm=%.2f  Ga=%.2f  LSTM=%.2f  →  R=%.3f\n', ...
        testCases{i, 1}, inputs(1), inputs(2), inputs(3), inputs(4), inputs(5), risk);
end

%% ── 8. 导出FIS ──
writeFIS(fis, 'WindTurbineFireRisk.fis');
fprintf('\n模糊推理系统已导出为 WindTurbineFireRisk.fis\n');
fprintf('可在MATLAB中键入 fuzzy(WindTurbineFireRisk) 打开FIS编辑器交互查看。\n');

disp(' ');
disp('===== 脚本 matlab_fuzzy_fis.m 运行完毕 =====');
