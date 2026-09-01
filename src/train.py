import os
import pandas as pd
from preprocessing import get_X_y
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

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

X_test , X_val , y_test , y_val = train_test_split(
    X_test ,
    y_test ,
    random_state = 42 ,
    stratify = y_test ,
    test_size = 0.5
)

X_train = torch.tensor(
    X_train.values ,
    dtype = torch.float32
)
y_train = torch.tensor(
    y_train.values ,
    dtype = torch.long
)
X_val = torch.tensor(
    X_val.values ,
    dtype = torch.float32
)
y_val = torch.tensor(
    y_val.values ,
    dtype = torch.long
)

model = nn.Sequential(

    nn.Linear(784 , 256) ,
    nn.ReLU() ,
    nn.Dropout(0.5) ,

    nn.Linear(256 , 128) ,
    nn.ReLU() ,
    nn.Dropout(0.5) ,

    nn.Linear(128 , 64) ,
    nn.ReLU() ,
    nn.Dropout(0.5) ,

    nn.Linear(64 , 16) ,
    nn.ReLU() ,
    nn.Dropout() ,

    nn.Linear(16 , 4) ,
    nn.ReLU() ,

    nn.Linear(16 , 10)
)