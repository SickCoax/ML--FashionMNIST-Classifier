def get_X_y(df) :

    X = df.drop(["label"] , axis = 1)
    y = df["label"]

    X = X / 255.0

    return X , y