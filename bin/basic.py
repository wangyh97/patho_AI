'''
basic functions


'''
import sys
import numpy as np
import torch
import cupy as cp



class Timer():
    def __init__(self,proc):
        self._begin_time = None
        self.proc = proc

    def tic(self):
        print(f'{self.proc} start!')
        self._begin_time = time.perf_counter()

    def toc(self):
        print(f'{self.proc} finish!,consuming {time.perf_counter() - self._begin_time}')
        return time.perf_counter() - self._begin_time

def binary_conversion(var):
    """
    二进制单位转换
    :param var: 需要计算的变量，bytes值
    :return: 单位转换后的变量，kb 或 mb
    """
    var_size = sys.getsizeof(var)
    assert isinstance(var_size, int)
    if var_size <= 1024:
        return f'占用 {round(var_size / 1024, 2)} KB内存'
    else:
        return f'占用 {round(var_size / (1024 ** 2), 2)} MB内存'
    
def show_info(show_OI = False,show_values = False,**kws):
    length = 'no length'
    keys = 'not a dict'
    values = 'not a dict'
    shape = 'not a ndarray'
    size = 'not a ndarray'
    for key,x in kws.items():
        if hasattr(x,'__len__'):
            length = len(x)
        if isinstance(x,dict):
            keys = x.keys()
            values = x.values()
        if type(x) is np.ndarray or type(x) is cp.ndarray or type(x) is torch.Tensor:
            shape = x.shape
            size = x.size
        print(key)
        print(f'allocated memory:{binary_conversion(x)}')
        print(f'\ntype:{type(x)} \nlen:{length}\nshape:{shape}\nsize:{size}\nkeys:{keys}\n')
        if show_OI:
            print(f'original info:{x}\n')
        if show_values:
            print(f'\nvalues:{values}\n')