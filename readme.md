## NeutronTP: Load-Balanced Distributed Full-Graph GNN Training with Tensor Parallelism. 

**NeutronTP** is a load-balanced and efficient distributed full-graph GNN training system with GNN tensor parallelism:

 * NeutronTP utilizes tensor parallelism for distributed GNN training, eliminating cross-worker vertex dependencies by partitioning features instead of graph structures.
 * NeutronTP employs a generalized decoupling training method to separate NN operations from graph aggregation operations, significantly reducing communication volume and frequency in GNN tensor parallelism.. 
 * NeutronTP employs a memory-efficient subgraph scheduling strategy to support large-scale graph processing and overlap the communication and computation tasks.


Currently, NeutronTP is under refactoring. We will release all features of NeutronOrch soon.

#### Getting started

1. Setup a clean environment.
```
"""
        推荐使用Anaconda
"""
conda create --name NTP
conda activate NTP
```
2. Install pytorch (needed for training) and other libraries (needed for downloading datasets). 


```
// Cuda 10:
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts
conda install -c dglteam dgl-cuda10.2
conda install pyg -c pyg -c conda-forge
pip install ogb
```

```
// Cuda 11:
首先使用以下指令安装包：
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
conda install -c dglteam dgl-cuda11.1
conda install pyg -c pyg -c conda-forge
pip install ogb
```

3. Compile and install spmm. (Optional. CUDA dev environment needed.)
```
cd spmm_cpp
python setup.py install
```

4. Prepare datasets (edit the code according to your needs).
```
//This may take a while.
python prepare_data.py
"""
        conda install "pytorch>=1.9" torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
        遇到pytorch版本问题，执行以上指令：存在pytorch版本过高，警告函数落后的风险
        conda remove pytorch torchvision torchaudio -y
        // 可以执行下面的指令
        conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 cudatoolkit=11.1 -c pytorch -c nvidia
"""
```
5. Train.
```

python main.py --nprocs=1 --nodes=16 --nlayers=2 --hidden=256 --epoch=100 --backend=nccl --dataset=reddit --model=GCN
        --nprocs | the number of GPUs in the node
        --nodes | the number of nodes
        --nlayers | the number of model layers
        --hidden | the dimension of hidden layers
        --nodes | the number of nodes
        --epoch | the number of epoch
        --backend | communication backend
        --dataset | input graph
        --model | training model
      
```


#### Contact

For the technical questions, please contact: **Xin Ai** (aixin0@stumail.neu.edu.cn) and **Hao Yuan** (yuanhao@stumail.neu.edu.cn)

