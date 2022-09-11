# Mac OS ERROR
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas

class CsvDataset(Dataset):

    def __init__(self, file_path=''):

        data_frame = pandas.read_csv(
            file_path, header=0, index_col=0,
            encoding='utf-8',
            skip_blank_lines=True
        )

        x_train = data_frame.iloc[:, :-1].values
        y_train = data_frame.iloc[:, -1].values

        self.x_train = torch.from_numpy(x_train)
        self.y_train = torch.from_numpy(y_train)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index]


if __name__ == '__main__':
    train_dataset = CsvDataset(file_path='./dataset/exampleForLUAD.csv')
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    for index, (batch_x, batch_y) in enumerate(train_loader):
        print(f'batch_id: {index}, {batch_x.shape}, {batch_y.shape}')
        print(batch_x, batch_y)

    # # 数据划分
    # train_dataset, validation_dataset = torch.utils.data.random_split(train_dataset, [5000, 1000])
    # train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    # validation_loader = DataLoader(validation_dataset, batch_size=8, shuffle=True)




        # # 训练模型
        # output = model(batch_x)
        # loss = criterion(output, batch_y)
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()