def get_X_y(df) :

    X = df.drop(["label"] , axis = 1)
    y = df["label"]

    return X , y