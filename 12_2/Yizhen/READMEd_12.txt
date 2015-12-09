Introduction for Dec.2

FUNCTION:
feature_FinelineNumber: extract Finlinenumber feature
MatchingMatrix_75feature: computing the correct rate for train samples with 75 features

MAT:
order_for_94247: the random order for the raw data, first 60000 is used as train, the rest is used as test.

result_10times: the 60000 train sample divided into training 30000 and validation 30000 and then create 38*38 mapping matrix. The row stands for the true label and column stands for the predicted label.

X_75_test: 34247*75 test samples
Y_75_test: labels for the 34247 test samples

X_75_train: 60000*75 test samples
Y_75_train: labels for the 60000 test samples

X_94247_1_5195: 94247*5195 first part
X_94247_2_5195: 94247*5195 second part
X_94247_3_5195: 94247*5195 third part

x_60000_5: type 5 training samples with 5195
x_60000_6: type 6 training samples with 5195
x_60000_7: type 7 training samples with 5195