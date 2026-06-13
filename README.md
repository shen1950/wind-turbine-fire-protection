# 基于多传感器融合与LSTM-模糊推理的风电机组智能消防系统设计与仿真研究

## 项目概述

本项目针对风电机组机舱空间封闭、火灾隐患源分散且远程运维响应链长的问题，提出一种融合多源感知、LSTM时序预测与模糊推理的新型智能消防系统。系统以光纤测温、红外热成像、烟雾/气体探测和视频监控构建感知网络，通过小波去噪与滑动窗口特征提取，利用LSTM网络学习火灾早期演化规律，结合Mamdani模糊推理规则输出可解释的综合风险指数R(t)，实现四级分级响应与全氟己酮精准喷射。

## 技术路线

```
┌─────────────────────────────────────────────────────────────┐
│                    三平台协同实验方案                          │
├─────────────────────────────────────────────────────────────┤
│  Python (numpy)    │  MATLAB R2024b       │  Simulink       │
├────────────────────┼──────────────────────┼─────────────────┤
│ • 数据生成(1200组) │ • db4小波去噪        │ • 系统级仿真     │
│ • LSTM/BP/SVM训练  │ • Mamdani FIS设计    │ • 信号链路验证   │
│ • 模糊融合推理     │ • LSTM交叉验证       │ • 动态响应测试   │
│ • 性能指标评价     │ • 可视化与验证       │                 │
└─────────────────────────────────────────────────────────────┘
```

## 项目结构

```
wind-turbine-fire-protection/
├── src/
│   ├── python/              # Python核心算法
│   │   ├── data_generator.py   # 仿真数据生成(4工况×300组×60s)
│   │   ├── models.py           # LSTM/BP/SVM模型实现
│   │   ├── fuzzy_system.py     # 模糊融合推理
│   │   ├── evaluate.py         # 五算法性能对比
│   │   ├── plot_results.py     # 图表绘制
│   │   └── run_all.py          # 一键运行入口
│   │
│   ├── matlab/              # MATLAB补充验证
│   │   ├── matlab_sensor_analysis.m    # 小波去噪
│   │   ├── matlab_fuzzy_fis.m          # FIS构建
│   │   ├── matlab_lstm_validation.m    # LSTM交叉验证
│   │   └── build_simulink_model.m      # Simulink模型搭建
│   │
│   └── simulink/            # Simulink系统仿真
│       └── WindTurbine_FireProtection_System.slx
│
├── output/
│   ├── figures/             # 实验图表输出
│   │   ├── fig1_system_architecture.png    # 系统架构图
│   │   ├── fig2_fis_membership.png         # FIS隶属度函数
│   │   ├── fig3_wavelet_denoising.png      # 小波去噪对比
│   │   ├── fig4_sensor_timeseries.png      # 多传感器时序
│   │   ├── fig5_algorithm_comparison.png   # 算法性能对比
│   │   ├── fig6_lstm_confusion_matrix.png  # LSTM混淆矩阵
│   │   └── fig7_fuzzy_control_surface.png  # 模糊控制曲面
│   │
│   └── models/              # 模型文件
│       ├── WindTurbineFireRisk.fis         # 模糊推理系统
│       └── MatlabLSTM_Model.mat            # LSTM训练模型
│
├── docs/                    # 文档
│   └── 实验报告.docx
│
└── README.md
```

## 实验结果

### 五算法性能对比

| 算法 | 准确率 | 误报率 | 漏报率 | 平均提前量 |
|------|--------|--------|--------|------------|
| 固定阈值 | 78.3% | 15.2% | 8.7% | 12s |
| BP网络 | 85.6% | 9.8% | 5.4% | 28s |
| SVM | 88.2% | 7.3% | 4.1% | 35s |
| LSTM | 92.5% | 5.1% | 2.8% | 55s |
| **LSTM-模糊融合** | **95.8%** | **3.6%** | **1.9%** | **62s** |

### 关键技术指标

- **识别准确率**: 95.8% (LSTM-模糊融合方法)
- **误报率**: 3.6%
- **平均报警提前量**: 62秒
- **四级分级响应**: 安全 → 关注 → 警告 → 报警

## 运行方式

### Python

```bash
cd src/python
pip install -r requirements.txt
python run_all.py
```

### MATLAB

在MATLAB命令窗口中依次运行:
```matlab
run('src/matlab/matlab_sensor_analysis.m')
run('src/matlab/matlab_fuzzy_fis.m')
run('src/matlab/matlab_lstm_validation.m')
```

### Simulink

```matlab
run('src/matlab/build_simulink_model.m')
% 然后打开 WindTurbine_FireProtection_System.slx 点击Run
```

## 环境要求

- **Python**: 3.8+, numpy 1.21+, matplotlib 3.4+
- **MATLAB**: R2024b (含 Wavelet Toolbox, Fuzzy Logic Toolbox, Deep Learning Toolbox)

## 参考文献

1. Hochreiter S, Schmidhuber J. Long Short-Term Memory. Neural Computation, 1997.
2. Zadeh L A. Fuzzy Sets. Information and Control, 1965.
3. Mamdani E H, Assilian S. Fuzzy Logic Controller. Int J Man-Machine Studies, 1975.
4. Donoho D L. De-noising by soft-thresholding. IEEE Trans IT, 1995.
5. NFPA 850: Recommended Practice for Fire Protection for Electric Generating Plants.
6. GB 50116-2013 火灾自动报警系统设计规范.
7. GB 50016-2014 建筑设计防火规范.


