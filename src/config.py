import os


# data related
## dirname获取__file__的当前目录，当__file__是相对路径运行时结果是空，所以需要加上realpath
## 若获取当前目录改成用getcwd函数，那么相对路径运行结果也是正确的，但是就需要修改vscode调试的当前目录（默认是用户根目录）
datasets = ['SMD', 'SMAP', 'MSL']   
pardir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) 
data_folder = os.path.join(pardir, "data")
processed_folder = os.path.join(pardir, "processed")

# model related
models = ['LOF', 'OneClassSVM', 'Transformer']
window_sz = 10
batch_sz = 128
epoch_num = 10