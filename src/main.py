import os
import pandas as pd
from train import train_model
from evaluate import get_scores
from preprocessing import get_X_y

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

model = train_model(
    X_train ,
    y_train
)

f1 , a = get_scores(
    model ,
    X_test ,
    y_test
)

print(f"F1 SCORE : {f1}")
print(f"ACCURACY : {a}")