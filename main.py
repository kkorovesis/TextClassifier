import time,pickle
import numpy as np
from features import generate_features
from evaluation import cross_validation, evaluate_testing
from curves import plot_learning_curve,plot_precision_recall


# Develop a text classifier for a kind of texts of your choice (e.g., e-mail messages, tweets, customer reviews) and at
# least two classes (e.g., spam/ham, positive/negative/neutral).2 You should write your own code to convert each
# (training, validation, or test) text to a feature vector. You may use Boolean, TF, or TF-IDF features corresponding
# to words or n-grams, to which you can also add other features (e.g., length of the text).3 You may apply any feature
# selection (or dimensionality reduction) method you consider appropriate. You may also want to try using centroids of
# pre-trained word embeddings (slide 35).4 You can write your own code to perform feature selection (or dimensionality
# reduction) and to train the classifier (e.g., using SGD and the tricks of slides 58 and 59, in the case of logistic
# regression), or you can use existing implementations.5 You should experiment with at least logistic regression, and
# optionally other learning algorithms (e.g., Naive Bayes, k-NN, SVM).
#
# Draw learning curves (slides 64, 67) with appropriate measures (e.g., accuracy, F1)
# and precision-recall curves(slide 23).
# Include experimental results of appropriate baselines (e.g., majority classifiers). Make sure that you
# use separate training and test data. Tune the feature set and hyper-parameters (e.g., regularization weight λ)
# on a held-out part of the training data or using a cross-validation (slide 25) on the training data. Document clearly
# in a short report (max. 10 pages) how your system works and its experimental results.


########################################## MAIN ##########################################

start_time = time.time()
print("--- %s seconds ---" % (time.time() - start_time), "\n")

# generate_features()

test_set = np.array(pickle.load( open( 'output_files\\feature_matrix_test.pkl', "rb" ) ))
train_set = np.array(pickle.load( open( 'output_files\\feature_matrix_train.pkl', "rb" ) ))
test_labels = np.array(pickle.load( open( 'output_files\\test_labels.pkl', "rb" ) ))
train_labels = np.array(pickle.load( open( 'output_files\\train_labels.pkl', "rb" ) ))

print("Training evaluation \n")
cross_validation(train_set, train_labels, "Logistic Regression Stochastic Gradient Descent", n_folds=10, n_jobs=4,pos_label=0)

print("Training evaluation \n")
cross_validation(train_set, train_labels, "k-NN", n_folds=10, n_jobs=4,pos_label=0)

print("Test Evaluation with Logistic Regression SGD \n")
evaluate_testing(train_set, train_labels ,test_set, test_labels, "Logistic Regression Stochastic Gradient Descent", pos_label=0)

print("Test Evaluation with Logistic Regression SGD \n")
evaluate_testing(train_set, train_labels ,test_set, test_labels, "k-NN", pos_label=0)

print("Test Evaluation with Random Forests \n")
evaluate_testing(train_set, train_labels ,test_set, test_labels, "Random Forests", pos_label=0)

# plot_precision_recall(train_set,train_labels,"Logistic Regression Stochastic Gradient Descent")
# plot_learning_curve(train_set,train_labels,"Logistic Regression Stochastic Gradient Descent")


print('############## TIME ################')
print("--- %s seconds ---" % (time.time() - start_time))
print('###############################'+'\n')
