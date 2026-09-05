import os
import pandas as pd
from preprocessing import get_X_y
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader , TensorDataset
from sklearn.metrics import f1_score , accuracy_score

csv_path = os.path.join(
    os.path.dirname(__file__) ,
    ".." ,
    "dataset" ,
    "fashion-mnist_train.csv"
)
df_train = pd.read_csv(csv_path)

csv_path = os.path.join(
    os.path.dirname(__file__) ,
    ".." ,
    "dataset" ,
    "fashion-mnist_test.csv"
)
df_test = pd.read_csv(csv_path)

X_train , y_train = get_X_y(df_train)
X_test , y_test = get_X_y(df_test)


X_train = X_train / 255.0
X_test = X_test / 255.0

X_train = torch.tensor(
    X_train.values ,
    dtype = torch.float32
)
y_train = torch.tensor(
    y_train.values ,
    dtype = torch.long
)


train_dataset = TensorDataset(
    X_train ,
    y_train
)

train_loader = DataLoader(
    train_dataset ,
    batch_size = 256 ,
    shuffle = True
)

model = nn.Sequential(

    nn.Linear(784 , 256) ,
    nn.LeakyReLU() ,
    nn.BatchNorm1d(256) ,
    nn.Dropout(0.03651566135425316) ,

    nn.Linear(256 , 128) ,
    nn.LeakyReLU() ,
    nn.BatchNorm1d(128) ,
    nn.Dropout(0.04011799029039411) ,

    nn.Linear(128 , 24) ,
    nn.LeakyReLU() ,
    nn.BatchNorm1d(24) ,
    nn.Dropout(0.03255362371798396) ,

    nn.Linear(24 , 24) ,
    nn.LeakyReLU() ,
    nn.BatchNorm1d(24) ,
    nn.Dropout(0.0) ,

    nn.Linear(24 , 10) 
)

def init_weight(m) : 

    if isinstance(m , nn.Linear) :
        nn.init.kaiming_normal(
            m.weight ,
            mode = "fan_in" ,
            nonlinearity = "leaky_relu"
        )
        nn.init.zeros_(m.bias)

model.apply(init_weight)

optimizer = optim.AdamW(
    model.parameters() ,
    lr = 0.004291440303963457 ,
    weight_decay = 4.415561505155581e-05
)

critrion = nn.CrossEntropyLoss()

for epoch in range(30) :

    model.train()

    for X_batch , y_batch in train_loader :

        optimizer.zero_grad()

        logits = model(X_batch) 

        loss = critrion(
            logits ,
            y_batch
        )

        loss.backward()

        optimizer.step()


X_test = torch.tensor(
    X_test.values ,
    dtype = torch.float32
)


model.eval()

with torch.no_grad():
    logits = model(X_test)

y_pred = torch.argmax(logits, dim=1)

y_pred = pd.Series(y_pred.numpy())

print(y_pred)


print(f1_score(y_test , y_pred , average = "macro"))

print(accuracy_score(y_test , y_pred))

