import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def fizzbuzz_encode(number):
    """
    number: int
    :param number:
    :return:
    """
    if number % 15 == 0:
        return 3 # fizzbuzz
    elif number % 5 == 0:
        return 2 # buzz
    elif number % 3 == 0:
        return 1 # fizz
    return 0 # str(number)

def fizzbuzz_decode(number, label):
    """
    number: int
    label: 0 1 2 3
    :param number:
    :param label:
    :return:
    """
    return [str(number), 'fizz', 'buzz', 'fizzbuzz'][label]

def helper(number):
    print(fizzbuzz_decode(number, fizzbuzz_encode(number)))

for number in range(1, 16):
    helper(number)

NUM_DIGITS = 10
def binary_encode(number):
    return np.array([number >> d & 1 for d in range(NUM_DIGITS)][::-1])
print(binary_encode(1))

x_train = torch.Tensor([binary_encode(number) for number in range(101, 1024)])
y_train = torch.LongTensor([fizzbuzz_encode(number) for number in range(101, 1024)])
print(x_train[:5])

class MyDataset(Data.Dataset):
    """
    """
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index]

    def __len__(self):
        return len(self.y_train)


train_dataset = MyDataset(x_train, y_train)
train_loader = Data.DataLoader(train_dataset, batch_size=16, shuffle=True)

class MyModel(nn.Module):
    def __init__(self, dim1, dim2):
        super(MyModel, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(10, dim1),
            nn.ReLU(),
            nn.Linear(dim1, dim2),
            nn.Linear(dim2, 4)
        )


    def forward(self, x):
        out = self.module(x)
        return out

model = MyModel(64, 8).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)

for k, v in model.named_parameters():
    print(k)

epoch = 1000
for i in range(epoch):
    for idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        if idx % 50 == 0:
            print(loss)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


x_test = torch.Tensor([binary_encode(15)]).to(device)
pred = model(x_test)
softmax = nn.Softmax()
pred = softmax(pred)
print(pred)
result = np.argmax(pred.cpu().detach().numpy(), 1)
print(result)




