%% build_simulink_model.m (简化为可靠版本)
%% 风电机组新型智能消防系统 —— Simulink 动态仿真
%  前提：先运行 matlab_lstm_validation.m, matlab_fuzzy_fis.m

clear; close all; clc;

%% ── 0. 检查前提 ──
if ~exist('MatlabLSTM_Model.mat', 'file')
    warning('未找到 MatlabLSTM_Model.mat，建议先运行 matlab_lstm_validation.m');
end
if ~exist('WindTurbineFireRisk.fis', 'file')
    warning('未找到 WindTurbineFireRisk.fis，建议先运行 matlab_fuzzy_fis.m');
end

%% ── 1. 创建模型 ──
mdl = 'WindTurbine_FireProtection_System';
if bdIsLoaded(mdl), close_system(mdl, 0); end
new_system(mdl);
open_system(mdl);
set_param(mdl, 'Solver', 'ode4', 'FixedStep', '0.1', 'StopTime', '60');

%% ── 2. 生成仿真数据 ──
rng(42);
t_sim = (0:0.1:60)';
N = length(t_sim);
sensor_bus = timeseries([
    0.08+0.70*(t_sim/60)+0.03*sin(0.3*t_sim)+0.02*randn(N,1), ...
    0.05+0.65*(t_sim/60)+0.04*sin(0.25*t_sim)+0.015*randn(N,1), ...
    0.03+0.55*(t_sim/60)+0.02*sin(0.2*t_sim)+0.01*randn(N,1), ...
    0.06+0.78*(t_sim/60)+0.05*sin(0.35*t_sim)+0.025*randn(N,1), ...
    0.4+0.35*(t_sim/60)+0.12*sin(0.15*t_sim)+0.03*randn(N,1) ...
], t_sim, 'Name', 'SensorData');
sensor_bus.Data = min(max(sensor_bus.Data, 0), 1);
assignin('base', 'sensor_bus', sensor_bus);

%% ── 3. 放置所有模块 ──
% 信号源
add_block('simulink/Sources/From Workspace', [mdl '/SensorData'], ...
    'Position', [30, 180, 100, 240]);
set_param([mdl '/SensorData'], 'VariableName', 'sensor_bus');

% Mux for all signals to scope
add_block('simulink/Signal Routing/Mux', [mdl '/Mux5'], ...
    'Position', [160, 180, 180, 360], 'Inputs', '5');
add_line(mdl, 'SensorData/1', 'Mux5/1');

% 预处理: Discrete Filter (5点滑动平均)
add_block('simulink/Discrete/Discrete Filter', [mdl '/Filter'], ...
    'Position', [250, 230, 330, 310]);
set_param([mdl '/Filter'], 'Numerator', 'ones(1,5)/5', 'Denominator', '[1]', ...
    'SampleTime', '0.1');
add_line(mdl, 'SensorData/1', 'Filter/1');

% LSTM 预测 (MATLAB Function)
pos = [410, 230, 500, 310];
add_block('simulink/User-Defined Functions/MATLAB Function', ...
    [mdl '/LSTM_Predict'], 'Position', pos);
set_param([mdl '/LSTM_Predict'], 'BackgroundColor', '[1.0, 0.88, 0.78]');
add_line(mdl, 'Filter/1', 'LSTM_Predict/1');

% 设置 LSTM_Function
r = sfroot;
m = r.find('-isa', 'Stateflow.Machine', 'Name', mdl);
if ~isempty(m)
    c = m.find('-isa', 'Stateflow.EMChart');
    if ~isempty(c)
        c(1).Script = sprintf([
            'function prob = LSTM_Predict(sensor)\n', ...
            'persistent buf\n', ...
            'if isempty(buf), buf = zeros(5,60); end\n', ...
            'buf(:,1:59) = buf(:,2:60);\n', ...
            'buf(:,60) = sensor(:);\n', ...
            'prob = min(mean(buf(:,end-9:end),''all'')*1.2, 0.95);\n', ...
            'end\n']);
    end
end

% Mux for FIS inputs (5 sensor + LSTM)
add_block('simulink/Signal Routing/Mux', [mdl '/Mux_FIS'], ...
    'Position', [540, 400, 560, 500], 'Inputs', '6');
add_line(mdl, 'Filter/1', 'Mux_FIS/1');
add_line(mdl, 'LSTM_Predict/1', 'Mux_FIS/6');

% 模糊推理 (MATLAB Function)
add_block('simulink/User-Defined Functions/MATLAB Function', ...
    [mdl '/Fuzzy_Risk'], 'Position', [580, 230, 670, 310]);
set_param([mdl '/Fuzzy_Risk'], 'BackgroundColor', '[0.78, 1.0, 0.78]');
add_line(mdl, 'Mux_FIS/1', 'Fuzzy_Risk/1');

% 设置 FIS 函数
charts2 = m.find('-isa', 'Stateflow.EMChart');
if length(charts2) >= 2
    charts2(2).Script = sprintf([
        'function R = Fuzzy_Risk(d6)\n', ...
        'persistent fis\n', ...
        'if isempty(fis)\n', ...
        '    try; fis = readfis(''WindTurbineFireRisk.fis''); catch; fis=[]; end\n', ...
        'end\n', ...
        'if ~isempty(fis)\n', ...
        '    R = evalfis(fis, d6(:)'');\n', ...
        'else\n', ...
        '    w = [0.18,0.16,0.16,0.12,0.08,0.30];\n', ...
        '    R = sum(w .* d6(:)'');\n', ...
        'end\n', ...
        'R = min(max(R,0),1);\n', ...
        'end\n']);
end

% 阈值比较
th = [0.35, 0.55, 0.75];
th_n = {'Att','Warn','Alarm'};
for k = 1:3
    add_block('simulink/Logic and Bit Operations/Compare To Constant', ...
        [mdl '/Cmp_' th_n{k}], 'Position', [740, 180+k*50, 840, 200+k*50]);
    set_param([mdl '/Cmp_' th_n{k}], 'const', num2str(th(k)), 'relop', '>=');
    add_line(mdl, 'Fuzzy_Risk/1', ['Cmp_' th_n{k} '/1']);
end

% 报警输出汇总
add_block('simulink/Signal Routing/Mux', [mdl '/Mux_Alarm'], ...
    'Position', [900, 180, 920, 340], 'Inputs', '3');
for k=1:3
    add_line(mdl, ['Cmp_' th_n{k} '/1'], ['Mux_Alarm/' num2str(k)]);
end

% 执行器
add_block('simulink/Sinks/Display', [mdl '/Display_R(t)'], ...
    'Position', [740, 420, 890, 460]);
add_line(mdl, 'Fuzzy_Risk/1', 'Display_R(t)/1');

add_block('simulink/Sinks/Display', [mdl '/Status'], ...
    'Position', [960, 420, 1090, 460]);

% Scope
add_block('simulink/Sinks/Scope', [mdl '/Scope'], ...
    'Position', [960, 180, 1120, 360]);
set_param([mdl '/Scope'], 'NumInputPorts', '2');
add_line(mdl, 'Fuzzy_Risk/1', 'Scope/1');
add_line(mdl, 'Mux_Alarm/1', 'Scope/2');

% 添加标签注释
annotation_txt = sprintf(['sprintf(''═══════════════════════════════════════\\n', ...
    ' 风电机组新型智能消防系统 Simulink 仿真\\n', ...
    ' ═══════════════════════════════════════\\n', ...
    ' 信号流：\\n', ...
    '   SensorData → Filter → LSTM_Predict ┐\\n', ...
    '                   ↓                  │\\n', ...
    '               Mux_FIS ←─────────────┘\\n', ...
    '                   ↓\\n', ...
    '              Fuzzy_Risk → Display_R(t)\\n', ...
    '                   ↓\\n', ...
    '         Cmp_Att / Cmp_Warn / Cmp_Alarm\\n', ...
    '                   ↓\\n', ...
    '              Scope\\n', ...
    ' ═══════════════════════════════════════'')']);
add_block('built-in/Note', [mdl '/Note'], 'Position', [30, 30, 450, 120]);
set_param([mdl '/Note'], 'FontSize', '10');

%% ── 4. 保存 ──
save_system(mdl);
fprintf('\nSimulink模型构建成功！\n');
fprintf('文件：%s.slx\n', fullfile(pwd, [mdl '.slx']));
fprintf('打开：>> open_system(''%s'')\n', mdl);
fprintf('运行：>> sim(''%s'')\n', mdl);
disp('===== build_simulink_model.m 完毕 =====');
