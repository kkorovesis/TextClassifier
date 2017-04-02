import  pickle
from sklearn.linear_model import LogisticRegression
from evaluation import print_metrics

test_set = pickle.load( open( 'output_files\\feature_matrix_test.pkl', "rb" ) )
train_set = pickle.load( open( 'output_files\\feature_matrix_train.pkl', "rb" ) )
test_labels = pickle.load( open( 'output_files\\test_labels.pkl', "rb" ) )
train_labels = pickle.load( open( 'output_files\\train_labels.pkl', "rb" ) )

lr = LogisticRegression(C=10)
lr.fit(train_set,train_labels)

predicted = lr.predict(test_set)
print_metrics(test_labels,predicted,0)

# Found: 2224  Ham mails  and  1022  Spam mails
# Found: 725  Ham mails  and  356  Spam mails