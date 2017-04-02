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