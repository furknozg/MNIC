
import data_classes as dconst
import mal_models as malmod
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import numpy as np
import ember
import pickle


# Import model
from sklearn.ensemble import IsolationForest

#SVM train
PATH = "./Models/ember2018/iforest.pickle"
DATA_PATH = "./Datasets/ember2018"


X_train, y_train,_,_ = ember.read_vectorized_features(DATA_PATH)
X_train = X_train[y_train == 1]


print("*"*50)
print("Training Isolation Forest")

forest = IsolationForest(n_estimators=100, max_samples='auto', contamination=0.001, 
                        max_features=.05, bootstrap=False, n_jobs=6, random_state=42, verbose=1).fit(X_train + np.random.normal(size=X_train.shape, scale=0.001))


print("Training Done")
print("*"*50)
# Save file as pickle
pickle.dump(forest, open(PATH, "wb"))

