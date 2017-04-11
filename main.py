import time,  pickle
import numpy as np
np.set_printoptions(threshold=np.inf)
from extractfeautures import load_data, tf_idf, tf_idf_testing

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

list_of_docs, test_list_of_docs , train_labels, test_labels = load_data()
train_feature_matrix,feature_names,idf = tf_idf(list_of_docs)

test_feature_matrix = tf_idf_testing(test_list_of_docs,idf,feature_names)

# Concatenate the train and test matrix to create unique feature matrix to do cross-validation

feature_matrix = train_feature_matrix + test_feature_matrix
labels = train_labels + test_labels

with open('output_files\\feature_matrix_train.pkl', "wb") as fp:
    pickle.dump(train_feature_matrix, fp)

with open('output_files\\train_labels.pkl', "wb") as fp:
    pickle.dump(train_labels, fp)

with open('output_files\\feature_matrix_test.pkl', "wb") as fp:
    pickle.dump(test_feature_matrix, fp)

with open('output_files\\test_labels.pkl', "wb") as fp:
    pickle.dump(test_labels, fp)

with open('output_files\\feature_matrix.pkl', "wb") as fp:
    pickle.dump(feature_matrix, fp)

with open('output_files\\labels.pkl', "wb") as fp:
    pickle.dump(labels, fp)


print('############## TIME ################')
print("--- %s seconds ---" % (time.time() - start_time))
print('###############################'+'\n')