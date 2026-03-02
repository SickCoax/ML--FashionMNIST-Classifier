# Fashion MNIST Classifier

This project is a machine learning model which can classify clothing images from the Fashion-MNIST dataset.

This model uses a Scikit-Learn pipeline with scaling , pca , and logreg

## Dataset

Fashion-MNIST contains 70000 grayscale images of clothing items.
- 60000 training images
- 10000 testing images
- Images size 28 x 28
- 784 pixel feature per image

Each row contains :
label , pixel1 , pixel2 , pixel3 ... pixel784

### Label Mapping
- 0 – T-shirt / Top
- 1 – Trouser
- 2 – Pullover
- 3 – Dress
- 4 – Coat
- 5 – Sandal
- 6 – Shirt
- 7 – Sneaker
- 8 – Bag
- 9 – Ankle Boot

### Note : This project uses the Fashion MNIST dataset. 

Download it from :
[Fashion MNIST Dataset](https://www.kaggle.com/datasets/zalando-research/fashionmnist)

After downloading, place the files inside "dataset" folder.


## Model Pipeline

StandardScaler ~> PCA ~> LogisticRegression

Hyperparameter tuning done using GridSearchCV