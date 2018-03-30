import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('../res/training.csv', delimiter='\t', header=None, low_memory=False, dtype=str)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(data[1], data[0])

vec = TfidfVectorizer()
X_train = vec.fit_transform(X_train_raw.values.astype('U'))
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

lines = pd.read_csv('../res/testing.csv', delimiter='\t', header=None, low_memory=False)

total = 0
true_spam = 0
false_spam = 0
true_ham = 0
false_ham = 0
for row in lines.itertuples():
    if isinstance(row[2], basestring):
        test = vec.transform([row[2].decode('utf-8', 'ignore').encode('utf-8')])
        prediction = classifier.predict(test)
        total += 1
        print '[Prediction: ' + prediction[0] + ']-> [Actual ' + row[1] + ']'
        if prediction[0] == row[1]:
            if prediction[0] == 'spam':
                true_spam += 1
            else:
                true_ham += 1
        else:
            if prediction[0] == 'spam':
                false_spam += 1
            else:
                false_ham += 1

print 'Logistic Regression Results'
print 'Training Data Size: ' + str(data.shape[0])
print 'Testing Data Size: ' + str(lines.shape[0])
print 'Correct Guesses: ' + str(total)
precision = true_ham / ((true_ham + false_ham) * 1.0)
recall = true_ham / ((true_ham + false_spam) * 1.0)
print 'Precision: ' + str(precision)
print 'Recall: ' + str(recall)
print 'Accuracy: ' + str((true_ham + true_spam) / ((true_ham + true_spam + false_ham + false_spam) * 1.0))
print 'F-Measure: ' + str(2 * ((precision * recall) / (precision + recall)))

