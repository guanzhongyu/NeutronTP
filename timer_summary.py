import pickle
import statistics
import os

def compute_final_timer_summary(timer_dir, world_size=2):
    """
    读取本地保存的每个 rank 的 duration_dict 文件，并计算平均和标准差 summary
    """
    all_durations = []
    count_dict = None  # 假设 count_dict 相同，取 rank0 的即可

    # 读取每个 rank 的文件
    for rank in range(world_size):
        filename = os.path.join(timer_dir, f"duration_dict_{rank}.pkl")
        if not os.path.exists(filename):
            print(f"[Rank {rank}] 文件 {filename} 不存在，跳过")
            continue

        with open(filename, "rb") as f:
            duration_dict = pickle.load(f)
            all_durations.append(duration_dict)

        # 取 rank0 的 count_dict
        if rank == 0:
            count_dict_file = filename  # 假设 count_dict 存在同名文件
            # 这里假设 count_dict 与 duration_dict 一起保存，如果单独保存，可修改
            count_dict = {k: 1 for k in duration_dict.keys()}  # 占位，如果有真实 count_dict，可加载

    if not all_durations:
        print("没有读取到任何 duration_dict 文件")
        return ""

    # 计算平均值和标准差
    avg_dict = {}
    std_dict = {}
    keys = all_durations[0].keys()
    for key in keys:
        data = [d[key] for d in all_durations]
        if len(data) == 1:
            avg_dict[key] = std_dict[key] = data[0]  # 单 GPU
        else:
            avg_dict[key] = statistics.mean(data)
            std_dict[key] = statistics.stdev(data)

    # 生成 summary 字符串
    s = '\ntimer summary:\n'
    for key in keys:
        s += "%6.2fs %6.2fs %5d %s\n" % (avg_dict[key], std_dict[key], count_dict.get(key, 0), key)

    return s


if __name__ == "__main__":
    # 本地保存的 timers 文件夹
    timer_dir = "./timers"
    summary = compute_final_timer_summary(timer_dir, world_size=2)
    print(summary)
    # 变量控制保存文件名
    file_prefix = "cora_1_timer_summary"  # 可以修改这个前缀
    file_name = f"{file_prefix}.txt"

    # 检查 ./result 文件夹是否存在，如果不存在则创建
    save_dir = "./result"
    os.makedirs(save_dir, exist_ok=True)

    # 拼接完整路径
    file_path = os.path.join(save_dir, file_name)

    # 保存 summary 到文件
    with open(file_path, "w") as f:
        f.write(summary)

    print(f"Summary saved to {file_path}")
