import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score

data = pd.read_csv('../res/mass.csv', delimiter=',', header=None, low_memory=False)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(data[1], data[0])

vec = TfidfVectorizer()
X_train = vec.fit_transform(X_train_raw)
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

X_test = vec.transform(["earn a 6 figure income online ! - 100 % automated system !"])

predict = classifier.predict(X_test)
print predict