import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_csv('../res/all.csv', delimiter='\t', header=None, low_memory=False, dtype=str)

x_train_raw, x_test_raw, y_train, y_test = train_test_split(data[1], data[0])

vec = TfidfVectorizer()
x_train = vec.fit_transform(x_train_raw.values.astype('U'))

classifier = MLPClassifier(max_iter=50)
classifier.fit(x_train, y_train)

print classifier
predictions = classifier.predict(vec.transform(x_test_raw.values.astype('U')))
print 'Confusion Matrix:'
print confusion_matrix(y_test, predictions)
print 'Report: '
print classification_report(y_test, predictions)
