import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader , TensorDataset


def train_model(X_train , y_train) :

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

    return model