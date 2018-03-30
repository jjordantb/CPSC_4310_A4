CPSC 4310 Assignment 4 README

Overview
- In this assignment we were asked to classify spam using logistic regression, a neural network, and SVM.
- We were provided with enron spam data
- I wrote a python script that extracts everything from each enron folder located in 'res'
    - The script dumps all of the data to 'all.csv'

Scripts - run with python 2
- Are located in 'src'
- 'logistic_regression.py' is the logistic regression
- 'neural_net.py' is the neural net
- 'svm.py' is the svm

Each was implemented with scikit-learn

Results, testing was done with the data that was obtained from the 'train_test_split' function
1. Logistic Regression
    - Worked well and worked fast, high levels of precision, recall
2. Neural Net
    - Worked about as well as logistic regression at 50 iterations, however those 50 iterations took about 100 times
        longer than logistic regression
3. SVM
    - Worked marginally better than the other two and was significantly faster than both, if the neural network
        ran for more iterations it might beat SVM
