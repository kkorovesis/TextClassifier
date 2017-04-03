import time,  pickle
import numpy as np
np.set_printoptions(threshold=np.inf)
from extractfeautures import load_data, tf_idf_list, tf_idf_list_testing

########################################## MAIN ##########################################
start_time = time.time()
print("--- %s seconds ---" % (time.time() - start_time), "\n")

list_of_docs, test_list_of_docs , train_labels, test_labels = load_data()
feature_matrix,feature_names,idf = tf_idf_list(list_of_docs)

test_feature_matrix = tf_idf_list_testing(test_list_of_docs,idf,feature_names)

with open('output_files\\feature_matrix_train.pkl', "wb") as fp:
    pickle.dump(feature_matrix, fp)

with open('output_files\\train_labels.pkl', "wb") as fp:
    pickle.dump(train_labels, fp)

with open('output_files\\feature_matrix_test.pkl', "wb") as fp:
    pickle.dump(test_feature_matrix, fp)

with open('output_files\\test_labels.pkl', "wb") as fp:
    pickle.dump(test_labels, fp)


print('############## TIME ################')
print("--- %s seconds ---" % (time.time() - start_time))
print('###############################'+'\n')