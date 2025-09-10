#!/bin/bash

# 从节点（node_rank=1）
ssh gzy@192.168.6.130 "source ~/anaconda3/bin/activate NTP && python ~/NeutronTP/main.py"

# 主节点（node_rank=0）
python ~/NeutronTP/main.py 
