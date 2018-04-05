CPSC 4310 Assignment 4 README

Overview
- In this assignment we were asked to classify spam using logistic regression, a neural network, and SVM.
- We were provided with enron spam data and I chose the pre-processed
- I further processed it down to just the subject of each email
- I wrote a python script that extracts everything from each enron folder located in 'res'
    - The script dumps all of the data to 'all.csv'

Scripts - run with python 2
- The lab computers have an outdated version of sklearn, in this version there is no neural network implementation... Please run with sklearn 0.19.1
- Are located in 'src'
- 'logistic_regression.py' is the logistic regression
- 'neural_net.py' is the neural net
- 'svm.py' is the svm

Each was implemented with scikit-learn
I had to use some deprecated functions in order for it to run on lab computers

Results, testing was done with the data that was obtained from the 'train_test_split' function
1. Logistic Regression
    - Worked well and worked fast, high levels of precision, recall
2. Neural Net
    - Worked about as well as logistic regression at 50 iterations, however those 50 iterations took about 100 times
        longer than logistic regression
3. SVM
    - Worked marginally better than the other two and was significantly faster than the neural net and about the same
        as logistic regression, if the neural network ran for more iterations it might beat SVM

Logistic Regression:
    Confusion Matrix:
    [[3775  349]
     [ 233 4015]]
    Report:
                 precision    recall  f1-score   support

            ham       0.94      0.92      0.93      4124
           spam       0.92      0.95      0.93      4248

    avg / total       0.93      0.93      0.93      8372

Neural Net:
    Confusion Matrix:
    [[3817  262]
     [ 289 4004]]
    Report:
                 precision    recall  f1-score   support

            ham       0.93      0.94      0.93      4079
           spam       0.94      0.93      0.94      4293

    avg / total       0.93      0.93      0.93      8372

SVM:
    Confusion Matrix:
    [[3970  269]
     [ 220 3913]]
    Report:
                 precision    recall  f1-score   support

            ham       0.95      0.94      0.94      4239
           spam       0.94      0.95      0.94      4133

    avg / total       0.94      0.94      0.94      8372
