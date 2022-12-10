import os
import time
import numpy as np
from config import datasets, processed_folder, models, window_sz, batch_sz
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import precision_score, recall_score, f1_score

def convert_to_windows(data):
    windows = []
    reshaped_data0 = data[0].reshape(1, data[0].shape[0])
    for i, _ in enumerate(data):
        if i >= window_sz:
            windows.append(torch.from_numpy(data[i-window_sz+1:i+1]))
        else:
            windows.append(torch.cat([torch.from_numpy(
                np.repeat(reshaped_data0, window_sz-i-1, axis=0)), torch.from_numpy(data[0:i+1])]))
    return torch.stack(windows)
    
def load_window_dataset(dataset):
    # 将数据集转换为窗口集
    windows = convert_to_windows(dataset)
    # print(windows.shape)
    # 将窗口集转为TensorDataset
    windows = torch.DoubleTensor(windows)
    window_dataset = TensorDataset(windows, windows)
    return window_dataset

def load_dataset(dataset):
    if dataset not in datasets:
        raise Exception(f'{dataset} not found. Check one of {datasets}')
    # 读取numpy文件
    file_prefix = ''
    if dataset == 'SMD':
        file_prefix = 'machine-1-1'
    elif dataset == 'SMAP':
        file_prefix = 'A-1'
    elif dataset == 'MSL':
        file_prefix = 'M-1'
    files = []
    for file_type in ['train', 'test', 'labels']:
        file_path = os.path.join(processed_folder, dataset, f'{file_prefix}_{file_type}.npy')
        files.append(np.load(file_path))
    # 返回train test labels
    return files[0], files[1], files[2]

def get_score_metrics(y_true, y_pred):
    print('precision: {}'.format(precision_score(y_true, y_pred, average='binary')))
    print('recall: {}'.format(recall_score(y_true, y_pred, average='binary')))
    print('f1 score: {}'.format(f1_score(y_true, y_pred, average='binary')))

def run_classic_model(model_name, train_d, test_d, labels):
    if model_name == 'OneClassSVM':
        # print(train_d.shape)
        start_time = time.time()
        svm_model = OneClassSVM(gamma='auto').fit(train_d)
        end_time = time.time()
        print(f'SVM train time: {end_time - start_time}s')
        start_time = time.time()
        y_pred = svm_model.predict(test_d)
        end_time = time.time()
        print(f'SVM predict time: {end_time - start_time}s')
        y_pred = [0 if pred == 1 else 1 for pred in y_pred]
        # results = y_pred == labels
        get_score_metrics(labels, y_pred)
    elif model_name == 'LOF':
        start_time = time.time()
        lof_model = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(train_d) # novelty=True才能对非训练集进行predict
        end_time = time.time()
        print(f'LOF train time: {end_time - start_time}s')
        start_time = time.time()
        y_pred = lof_model.predict(test_d)
        end_time = time.time()
        print(f'SVM predict time: {end_time - start_time}s')
        y_pred = [0 if pred == 1 else 1 for pred in y_pred]
        # results = y_pred == labels
        get_score_metrics(labels, y_pred)

if __name__ == '__main__':
    # choose dataset and model 后面可改成通过命令行参数的形式传入
    input_model = 'LOF'  
    dataset = 'SMD'
    # load dataset 
    train_d, test_d, labels = load_dataset(dataset)
    # check model
    if input_model not in models:
        raise Exception(f'{input_model} is not implemented in {models}')
    # run model on dataset
    if input_model != 'Transformer':
        run_classic_model(input_model, train_d, test_d, labels)
    
    # # process dataset with Dataloader
    # train_loader = DataLoader(train_d, batch_size=batch_sz, shuffle=True)
    # test_loader = DataLoader(test_d, batch_size=batch_sz) # test就不shuffle了，因为需要每次窗口取最后一个检测值积累到输出，这是有顺序要求的
    