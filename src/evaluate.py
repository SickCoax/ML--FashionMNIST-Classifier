import torch
import pandas as pd
from sklearn.metrics import f1_score , accuracy_score

def get_scores(model , X_test , y_test) :

    X_test = torch.tensor(
        X_test.values ,
        dtype = torch.float32
    )

    model.eval()

    with torch.no_grad() :
        y_pred = model(X_test)

    y_pred = torch.argmax(y_pred , dim=1)
    y_pred = pd.Series(y_pred.numpy())

    f1 = f1_score(
        y_test ,
        y_pred ,
        average = "macro"
    )

    a = accuracy_score(
        y_test ,
        y_pred
    )

    return f1 , a