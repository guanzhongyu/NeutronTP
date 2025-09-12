import matplotlib.pyplot as plt
import numpy as np

# ======== 可修改数据区 ========

# --- COM ---
COM_1_gpu = 9924.0
COM_2_gpu = 5614.1
COM_1_time = 12.18
COM_2_time = 946.34

# --- cora ---
cora_1_gpu = 652.0
cora_2_gpu = 510.0
cora_1_time = 2.67
cora_2_time = 6.54

# --- LJ ---
LJ_1_gpu = 7296.0
LJ_2_gpu = 3856.0
LJ_1_time = 8.39
LJ_2_time = 603.03

# 输出图片文件名
gpu_fig_name = "gpu_usage_comparison.png"
time_fig_name = "time_comparison.png"

# ======== 绘图逻辑区 ========

# 数据整理
datasets = ["COM", "cora", "LJ"]
gpu_values = [
    [COM_1_gpu, COM_2_gpu],
    [cora_1_gpu, cora_2_gpu],
    [LJ_1_gpu, LJ_2_gpu]
]
time_values = [
    [COM_1_time, COM_2_time],
    [cora_1_time, cora_2_time],
    [LJ_1_time, LJ_2_time]
]

# 中文字体支持可去掉
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

x = np.arange(len(datasets))
width = 0.35

# ---------- GPU Memory Usage ----------
fig1, ax1 = plt.subplots(figsize=(6,4))
ax1.bar(x - width/2, [row[0] for row in gpu_values], width, label='1 Node')
ax1.bar(x + width/2, [row[1] for row in gpu_values], width, label='2 Nodes')
ax1.set_ylabel('Average GPU Memory (MB)')
ax1.set_title('GPU Memory Usage for Different Datasets')
ax1.set_xticks(x)
ax1.set_xticklabels(datasets)
ax1.legend()
plt.tight_layout()
plt.savefig(gpu_fig_name, dpi=300)

# ---------- Runtime ----------
fig2, ax2 = plt.subplots(figsize=(6,4))
ax2.bar(x - width/2, [row[0] for row in time_values], width, label='1 Node')
ax2.bar(x + width/2, [row[1] for row in time_values], width, label='2 Nodes')
ax2.set_ylabel('Total Runtime (s)')
ax2.set_title('Runtime for Different Datasets')
ax2.set_xticks(x)
ax2.set_xticklabels(datasets)
ax2.legend()
plt.tight_layout()
plt.savefig(time_fig_name, dpi=300)

print(f"Figures saved as {gpu_fig_name} and {time_fig_name}")
