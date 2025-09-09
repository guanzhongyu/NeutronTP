import torch
from setuptools import setup, Extension
from torch.utils import cpp_extension

from torch.utils.cpp_extension import BuildExtension, CUDAExtension


# ext_name = 'spmm_cpp'
# cpp_src = 'spmm.cpp' 
ext_name = 'spmm_cpp'
cpp_src = 'spmm.cpp' 

if torch.version.cuda.startswith('10'):
    cpp_src = 'spmm_original.cpp' 

# 将 cpp_extension.CppExtension 更改为 cpp_extension.CUDAExtension
"""
    这样 PyTorch 就会知道这个扩展需要用 nvcc 来编译，并且会自动处理 CUDA 相关的头文件和库路径
"""
setup(name=ext_name, 
        ext_modules=[cpp_extension.CUDAExtension(ext_name, [cpp_src])], 
        cmdclass={'build_ext': cpp_extension.BuildExtension})
