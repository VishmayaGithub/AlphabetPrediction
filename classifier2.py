import numpy as np
import pandas as pd
from scipy.sparse.construct import random
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from PIL import Image
import PIL.ImageOps

X = np.load('image.npz')['arr_0']
y = pd.read_csv("labels.csv")["labels"]
print(pd.Series(y).value_counts())
classes = ['A', 'B', 'C', 'D', 'E','F', 'G', 'H', 'I', 'J', "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
nclasses = len(classes)
x_train,x_test,y_train,y_test = train_test_split(X,y,random_state=9,test_size=2500,train_size=7500)
x_train_scaled = x_train/255.0
x_test_scaled = x_test/255.0

clf = LogisticRegression(solver='saga',multi_class="multinomial").fit(x_train_scaled,y_train)

def getPrediction(image):
    impil = Image.open(image)
    scalar = impil.convert('L')
    resize = scalar.resize((28,28),Image.ANTIALIAS)
    pixel_filter = 20
    min_pixel = np.percentile(resize, pixel_filter)
    image_bw_resized_inverted_scaled = np.clip(resize-min_pixel, 0, 255)
    max_pixel = np.max(resize)
    image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel
    test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)
    y_pred = clf.predict(test_sample)
    return y_pred[0] 
