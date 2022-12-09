import os
import sys
import pandas as pd
import numpy as np
import pickle
import json
from shutil import copyfile

datasets = ['SMD', 'SMAP', 'MSL']   
# dirname获取__file__的当前目录，当__file__是相对路径运行时结果是空，所以需要加上realpath
# 若获取当前目录改成用getcwd函数，那么相对路径运行结果也是正确的，但是就需要修改vscode调试的当前目录（默认是用户根目录）
pardir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) 
data_folder = os.path.join(pardir, "data")
output_folder = os.path.join(pardir, "processed")

# 加载并转存SMD数据集的train和test数据
def load_and_save(category, filename, dataset, dataset_folder):
    temp = np.genfromtxt(os.path.join(dataset_folder, category, filename),
                         dtype=np.float64,
                         delimiter=',')
    print(dataset, category, filename, temp.shape)
    np.save(os.path.join(output_folder, f"SMD/{dataset}_{category}.npy"), temp)
    return temp.shape

# 加载并转存SMD数据集的label数据
def load_and_save_label(category, filename, dataset, dataset_folder):
    temp = np.genfromtxt(os.path.join(dataset_folder, category, filename), dtype=np.uint8)
    print(dataset, category, filename, temp.shape)
    np.save(os.path.join(output_folder, f"SMD/{dataset}_{category}.npy"), temp)

# 最大最小归一化
def normalize(a, min_a = None, max_a = None):
	if min_a is None: min_a, max_a = np.min(a, axis = 0), np.max(a, axis = 0)
	return (a - min_a) / (max_a - min_a + 0.0001), min_a, max_a


def load_data(dataset):
	if dataset in datasets:
		folder = os.path.join(output_folder, dataset)
		os.makedirs(folder, exist_ok=True) # 创建输出结果所在文件夹
	if dataset == 'SMD':
		dataset_folder = os.path.join(data_folder, 'SMD')
		file_list = os.listdir(os.path.join(dataset_folder, "train"))
		for filename in file_list:
			if filename.endswith('.txt'):
				load_and_save('train', filename, filename.strip('.txt'), dataset_folder)
				load_and_save('test', filename, filename.strip('.txt'), dataset_folder)
				load_and_save_label('labels', filename, filename.strip('.txt'), dataset_folder)
	elif dataset in ['SMAP', 'MSL']:
		dataset_folder = os.path.join(data_folder, 'SMAP_MSL')
		file = os.path.join(dataset_folder, 'labeled_anomalies.csv')
		values = pd.read_csv(file)
		values = values[values['spacecraft'] == dataset] # 筛选对应数据集的行数
		filenames = values['chan_id'].values.tolist()  # 获得chan_id 
		for fn in filenames:
			# 打开对应文件，将训练集和测试集最大最小归一化
			train = np.load(f'{dataset_folder}/train/{fn}.npy') 
			test = np.load(f'{dataset_folder}/test/{fn}.npy')
			train, min_a, max_a = normalize(train)  
			test, _, _ = normalize(test, min_a, max_a)  # 用训练集的最大最小值，对测试集归一化
			np.save(f'{folder}/{fn}_train.npy', train)
			np.save(f'{folder}/{fn}_test.npy', test)
			# 给测试文件的异常点打标签
			labels = np.zeros(test.shape)
			 # 筛选出当前文件中异常点的下标 ['anomaly_sequences']取出是series，values转array 取第一项（也就一项）
			indices = values[values['chan_id'] == fn]['anomaly_sequences'].values[0]
			# "[2149, 2349], [4536, 4844], [3539, 3779]"   下面的做法就是转为数字序列，每两个就是一个区间，对应区间打标签
			indices = indices.replace(']', '').replace('[', '').split(', ')
			indices = [int(i) for i in indices]
			for i in range(0, len(indices), 2):
				labels[indices[i]:indices[i+1], :] = 1
			np.save(f'{folder}/{fn}_labels.npy', labels)
	else:
		raise Exception(f'{dataset} not found. Check one of {datasets}')

if __name__ == '__main__':
    input_datasets = sys.argv[1:]
    print(input_datasets, data_folder, output_folder)
    if len(input_datasets) > 0:
        for dataset in input_datasets:
            load_data(dataset)
    else:
        print("Usage: python preprocess.py <datasets>")
        print("where <datasets> is a space separated list of dataset names")