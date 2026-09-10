def get_X_y(df) :

    X = df.drop(["label"] , axis = 1)
    y = df["label"]

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    return X , y