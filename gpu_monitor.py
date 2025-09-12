import pickle
import os
import time
import subprocess
import statistics

class GPUMemoryMonitorLocal:
    def __init__(self, rank=0):
        self.rank = rank  # 用于文件命名
        self.memory_stats = []  # 存储内存监控数据
        self.gpu_recorder_dir = "./gpu_recorder"
        os.makedirs(self.gpu_recorder_dir, exist_ok=True)

    def record_epoch_memory(self):
        """记录当前GPU内存信息一次（用于每个epoch调用一次）"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,nounits"],
                capture_output=True, text=True, timeout=5
            )

            if result.returncode == 0:
                mem_used, mem_total = result.stdout.strip().split('\n')[1].split(',')
                mem_used = int(mem_used.strip())
                mem_total = int(mem_total.strip())
                mem_percent = (mem_used / mem_total) * 100

                self.memory_stats.append({
                    'timestamp': time.time(),
                    'mem_used_mb': mem_used,
                    'mem_total_mb': mem_total,
                    'mem_percent': mem_percent
                })
        except Exception as e:
            print(f"[Rank {self.rank}] GPU内存记录错误: {e}")

    def save_memory_stats(self):
        """把本地内存数据保存到文件夹"""
        file_path = os.path.join(self.gpu_recorder_dir, f"gpu_memory_{self.rank}.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(self.memory_stats, f)
        print(f"[Rank {self.rank}] GPU内存数据已保存到 {file_path}")

    def compute_summary(self):
        """计算平均值和峰值"""
        if not self.memory_stats:
            return None

        mem_percents = [s['mem_percent'] for s in self.memory_stats]
        mem_used_mb = [s['mem_used_mb'] for s in self.memory_stats]

        summary = {
            'avg_mem_percent': statistics.mean(mem_percents),
            'max_mem_percent': max(mem_percents),
            'avg_mem_used_mb': statistics.mean(mem_used_mb),
            'max_mem_used_mb': max(mem_used_mb),
            'sample_count': len(self.memory_stats)
        }
        return summary

    def print_summary(self, summary):
        if not summary:
            print("没有 GPU 内存数据")
            return

        print("\n" + "="*60)
        print("GPU内存使用情况汇总报告（本地）")
        print("="*60)
        print(f"平均内存使用率: {summary['avg_mem_percent']:.1f}%")
        print(f"峰值内存使用率: {summary['max_mem_percent']:.1f}%")
        print(f"平均内存使用量: {summary['avg_mem_used_mb']:.1f} MB")
        print(f"峰值内存使用量: {summary['max_mem_used_mb']:.1f} MB")
        print(f"样本数: {summary['sample_count']}")
        print("="*60 + "\n")
