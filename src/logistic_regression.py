import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_csv('../res/all.csv', delimiter='\t', header=None, low_memory=False, dtype=str)

x_train, x_test, y_train, y_test = train_test_split(data[1], data[0])

vec = TfidfVectorizer()
X_train = vec.fit_transform(x_train.values.astype('U'))
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

predictions = classifier.predict(vec.transform(x_test.values.astype('U')))
print 'Confusion Matrix:'
print confusion_matrix(y_test, predictions)
print 'Report: '
print classification_report(y_test, predictions)

