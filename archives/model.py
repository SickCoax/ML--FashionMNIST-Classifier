import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA


train_df = pd.read_csv("dataset/fashion-mnist_train.csv")

test_df = pd.read_csv("dataset/fashion-mnist_test.csv")

xtrain = train_df.drop(["label"] , axis=1)
ytrain = train_df["label"]

xtest = test_df.drop(["label"] , axis=1)    
ytest = test_df["label"]

pipeline = Pipeline([("scale" , StandardScaler()) ,("pca" , PCA(n_components=50)) ,("logreg" , LogisticRegression(max_iter=10000))])

param_grid = {"pca__n_components" : [50,100,150] , "logreg__C" : [0.01 , 0.1 , 1 ,10] , "logreg__solver" : ["lbfgs" , "saga"]}

grid = GridSearchCV(pipeline , param_grid , cv=5 , scoring="f1_weighted" , n_jobs=-1)
grid.fit(xtrain , ytrain)
model = grid.best_estimator_

ypred = model.predict(xtest)
print(ypred)
print()


# ----------------------
# Evaluation Matric
# ----------------------
print(classification_report(ytest , ypred))