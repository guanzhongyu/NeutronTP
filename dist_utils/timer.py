import datetime as dt
import torch
import math
import time
import pickle
import statistics
from collections import defaultdict


class TimerCtx:
    def __init__(self, timer, key, cuda):
        self.cuda = cuda
        self.timer = timer
        self.key = key

    def __enter__(self):
        if self.cuda:
            torch.cuda.synchronize()
        self.timer.start_time_dict[self.key] = time.time()
        return self

    def __exit__(self, type, value, traceback):
        if self.cuda:
            torch.cuda.synchronize()
        d = time.time() - self.timer.start_time_dict[self.key]
        self.timer.duration_dict[self.key] += d
        self.timer.count_dict[self.key] += 1


class DistTimer:
    def __init__(self, env):
        self.env = env
        self.start_time_dict = {}
        self.duration_dict = defaultdict(float)
        self.count_dict = defaultdict(int)

    def summary(self):
        s = '\ntimer summary:\n' +  "\n".join("%6.2fs %5d %s" % (self.duration_dict[key], self.count_dict[key], key) for key in self.duration_dict)
        return s

    def sync_duration_dicts(self):
        print(f"[{self.env.rank}] Before writing duration_dict")  # 调试
        self.env.store.set('duration_dict_%d'%self.env.rank, pickle.dumps(self.duration_dict))
        print(f"[{self.env.rank}] Finished writing duration_dict, entering barrier")  # 调试
        
        self.env.barrier_all()
        print(f"[{self.env.rank}] Passed barrier")  # 调试

        if self.env.rank == 0:
            """# 先单独尝试读取Rank 1的数据（关键调试步骤）
            
            rank1_data = pickle.loads(self.env.store.get('duration_dict_1'))
            print(f"[Rank 0] DEBUG: Successfully read Rank 1 data! Keys: {rank1_data.keys()}")  # 明确输出读取成功"""

            self.all_durations = [pickle.loads(self.env.store.get('duration_dict_%d'%rank)) for rank in range(self.env.world_size)]
            print(f"[{self.env.rank}] Finished reading all duration_dicts")  # 调试

    def summary_all(self):
        print(f"[{self.env.rank}] Calling sync_duration_dicts()")
        self.sync_duration_dicts()
        print(f"[{self.env.rank}] Returned from sync_duration_dicts()")
        
        if self.env.rank != 0:
            print(f"[{self.env.rank}] Not rank 0, returning None")
            return None

        print(f"[{self.env.rank}] Computing summary")

        # rank 0 计算平均值和标准差
        avg_dict = {}
        std_dict = {}
        for key in self.duration_dict:
            data = [d[key] for d in self.all_durations]
            if len(data) == 1:
                avg_dict[key], std_dict[key] = data[0], data[0]  #single GPU
            else:
                avg_dict[key], std_dict[key] = statistics.mean(data), statistics.stdev(data)       
        s = '\ntimer summary:\n' +  "\n".join("%6.2fs %6.2fs %5d %s" % (avg_dict[key], std_dict[key], self.count_dict[key], key) for key in self.duration_dict)
        return s

    def detail_all(self):
        self.sync_duration_dicts()
        avg_dict = {}
        std_dict = {}
        detail_dict = {}
        for key in self.duration_dict:
            data = [d[key] for d in self.all_durations]
            if len(data) == 1:
                avg_dict[key], std_dict[key] = data[0], data[0]  #single GPU
            else:
                avg_dict[key], std_dict[key] = statistics.mean(data), statistics.stdev(data)  
            detail_dict[key] = ' '.join("%6.2f"%x for x in data)
        s = '\ntimer summary:\n' +  "\n".join("%6.2fs %6.2fs %5d %s \ndetail: %s \n--------------" % (avg_dict[key], std_dict[key], self.count_dict[key], key, detail_dict[key]) for key in self.duration_dict)
        return s

    def timing(self, key):
        return TimerCtx(self, key, cuda=False)

    def timing_cuda(self, key):
        return TimerCtx(self, key, cuda=True)

    def start(self, key):
        self.start_time_dict[key] = time.time()
        return self.start_time_dict[key]

    def stop(self, key, *other_keys):
        def log(k, d=time.time() - self.start_time_dict[key]):
            self.duration_dict[k]+=d
            self.count_dict[k]+=1
        log(key)
        for subkey in other_keys:
            log(key+'-'+subkey)
        return


if __name__ == '__main__':
    pass

