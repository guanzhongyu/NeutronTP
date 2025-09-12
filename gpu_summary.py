import pickle
import statistics
import os

def compute_final_gpu_summary(gpu_dir, world_size=2):
    """
    读取本地保存的每个 rank 的 GPU 内存数据文件，并计算平均和最大值 summary
    """
    all_stats = []

    # 读取每个 rank 的文件
    for rank in range(world_size):
        filename = os.path.join(gpu_dir, f"gpu_memory_{rank}.pkl")
        if not os.path.exists(filename):
            print(f"[Rank {rank}] 文件 {filename} 不存在，跳过")
            continue

        with open(filename, "rb") as f:
            stats = pickle.load(f)
            all_stats.append(stats)

    if not all_stats:
        print("没有读取到任何 GPU 内存数据文件")
        return ""

    # 汇总每个 rank 的平均和峰值
    summary_lines = []
    overall_avg_percent = []
    overall_max_percent = []
    overall_avg_mb = []
    overall_max_mb = []

    for rank, stats in enumerate(all_stats):
        if not stats:
            summary_lines.append(f"Rank {rank}: No data")
            continue

        percents = [s['mem_percent'] for s in stats]
        used_mb = [s['mem_used_mb'] for s in stats]

        avg_percent = statistics.mean(percents)
        max_percent = max(percents)
        avg_mb = statistics.mean(used_mb)
        max_mb = max(used_mb)

        overall_avg_percent.append(avg_percent)
        overall_max_percent.append(max_percent)
        overall_avg_mb.append(avg_mb)
        overall_max_mb.append(max_mb)

        summary_lines.append(
            f"Rank {rank}: 平均 {avg_percent:.1f}% ({avg_mb:.1f} MB), "
            f"峰值 {max_percent:.1f}% ({max_mb:.1f} MB), 样本数 {len(stats)}"
        )

    # 汇总全局
    global_avg_percent = statistics.mean(overall_avg_percent) if overall_avg_percent else 0
    global_max_percent = max(overall_max_percent) if overall_max_percent else 0
    global_avg_mb = statistics.mean(overall_avg_mb) if overall_avg_mb else 0
    global_max_mb = max(overall_max_mb) if overall_max_mb else 0

    summary_str = "\nGPU Memory Summary:\n"
    summary_str += "="*60 + "\n"
    summary_str += f"全局平均内存使用率: {global_avg_percent:.1f}%\n"
    summary_str += f"全局峰值内存使用率: {global_max_percent:.1f}%\n"
    summary_str += f"全局平均内存使用量: {global_avg_mb:.1f} MB\n"
    summary_str += f"全局峰值内存使用量: {global_max_mb:.1f} MB\n"
    summary_str += "-"*60 + "\n"
    summary_str += "\n".join(summary_lines)
    summary_str += "\n" + "="*60 + "\n"

    return summary_str


if __name__ == "__main__":
    # 本地保存的 gpu_recorder 文件夹
    gpu_dir = "./gpu_recorder"
    summary = compute_final_gpu_summary(gpu_dir, world_size=2)
    print(summary)

    # 文件名由用户指定
    file_prefix = "cora_1_gpu_summary"  # 可修改
    file_name = f"{file_prefix}.txt"

    # 检查 ./result 文件夹是否存在
    save_dir = "./result"
    os.makedirs(save_dir, exist_ok=True)

    file_path = os.path.join(save_dir, file_name)

    # 保存 summary 到文件
    with open(file_path, "w") as f:
        f.write(summary)

    print(f"GPU Summary saved to {file_path}")
