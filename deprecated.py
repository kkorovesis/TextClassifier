# def reduce_features(features,nof): # returns the list of features with the highest tf_idf score of all docs
#
#     sum_list = ([float("{0:.2f}".format(sum(x))) for x in zip(*features)])
#     enum_sorted = (sorted((e, i) for i, e in enumerate(sum_list)))
#     first_enum_sorted = enum_sorted[-nof:]
#     best_features = [i for e, i in first_enum_sorted]
#
#     doc_best_features = []
#     for doc in features:
#         doc = [doc[i] for i in best_features]
#         doc_best_features.append(doc)
#
#     return doc_best_features


######################################################################################
'''TERM FREQUENCY WITH COUNTVECTORIZER'''

# tf = CountVectorizer( min_df=0, tokenizer=word_tokenize, max_features=200)
#
# results = tf.fit_transform(list_of_docs).toarray()
# names = tf.get_feature_names()
#
# for record in results:
#     for n,r in zip(names,record):
#         print(n,r)
#     exit()
######################################################################################
######################################################################################
# ''' REDUCED FEATURES USING LISTS '''

# for doc in tfidf_results:
#     n = int(math.ceil(len(doc)/3))
#
# tfidf_reduced = reduce_features(tfidf_results,n)
#
# for doc in tfidf_results:
#     print("Number of features : ", len(doc))
#     print(doc)
#
# for doc in tfidf_reduced:
#     print("Number of features : ", len(doc))
#     print(doc)
######################################################################################


#
# def print_precision_recall_curve(test_labels,predicted):
#
#     precision = dict()
#     recall = dict()
#     precision[0], recall[0], _ = precision_recall_curve(test_labels, predicted, 1)  # SPAM #
#     precision[1], recall[1], _ = precision_recall_curve(test_labels, predicted, 0)  # HAM #
#
#     # Plot Precision-Recall curve
#     lw = 2
#     plt.clf()
#     plt.plot(recall[0], precision[0], lw=lw, color='navy', label='Precision-Recall curve')
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.ylim([0.0, 1.05])
#     plt.xlim([0.0, 1.0])
#     # plt.yticks([i / 100.0 for i in range(0, 1)])
#     # plt.xticks([i / 100.0 for i in range(0, 1)])
#     plt.title('Precision-Recall Class: SPAM')
#     plt.legend(loc="lower left")
#     plt.show()
#
#     lw = 2
#     plt.clf()
#     plt.plot(recall[1], precision[1], lw=lw, color='turquoise', label='Precision-Recall curve')
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.ylim([0.0, 1.05])
#     plt.xlim([0.0, 1.0])
#     # plt.yticks([i / 10.0 for i in range(0, 1)])
#     # plt.xticks([i / 10.0 for i in range(0, 1)])
#     plt.title('Precision-Recall Class: HAM')
#     plt.legend(loc="lower left")
#     plt.show()
#
# def plot_precision_recall(test_labels,predicted, n_folds=10, n_jobs=4):
#
#     mean_recall = np.linspace(0, 1, 10)
#     reversed_mean_precision = 0.0
#     kf = KFold(len(test_labels), shuffle=True, n_folds=n_folds)
#
#     precision, recall, _ = precision_recall_curve(test_labels, predicted)
#
#     reversed_recall = np.fliplr([recall])[0]
#     reversed_precision = np.fliplr([precision])[0]
#
#     reversed_mean_precision += interp(mean_recall, reversed_recall, reversed_precision)
#     reversed_mean_precision[-1] = 0.0
#
#     reversed_mean_precision /= n_folds
#     reversed_mean_precision[0] = 1.0
#
#     mean_auc_pr = auc(mean_recall, reversed_mean_precision)
#     plt.plot(mean_recall, reversed_mean_precision, label='%s (area = %0.2f)' % (" LG-SAG ", mean_auc_pr), lw=1)
#     plt.xlim([-0.05, 1.05])
#     plt.ylim([-0.05, 1.05])
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.title('Precision Recall')
#     plt.legend(loc="lower right")
#     plt.show()