# Develop a text classifier for a kind of texts of your choice (e.g., e-mail messages, tweets, customer reviews) and
# at least two classes (e.g., spam/ham, positive/negative/neutral).2 You should write your own code to convert each
# (training, validation, or test) text to a feature vector. You may use Boolean, TF, or TF-IDF features corresponding
# to words or n-grams, to which you can also add other features (e.g., length of the text).3 You may apply any feature
# selection (or dimensionality reduction) method you consider appropriate. You may also want to try using centroids of
# pre-trained word embeddings (slide 35).4 You can write your own code to perform feature selection (or dimensionality
# reduction) and to train the classifier (e.g., using SGD and the tricks of slides 58 and 59, in the case of logistic
# regression), or you can use existing implementations.5 You should experiment with at least logistic regression, and
# optionally other learning algorithms (e.g., Naive Bayes, k-NN, SVM). Draw learning curves (slides 64, 67) with
# appropriate measures (e.g., accuracy, F1) and precision-recall curves (slide 23). Include experimental results
# of appropriate baselines (e.g., majority classifiers). Make sure that you use separate training and test data.
# Tune the feature set and hyper-parameters (e.g., regularization weight λ) on a held-out part of the training data
# or using a cross-validation (slide 25) on the training data. Document clearly in a short report (max. 10 pages)
# how your system works and its experimental results.

import nltk ,re
from enronparse import email_body,email_label
from tools import remove_punc
from nltk.stem.snowball import SnowballStemmer

def calculateUnigramProbLS(unigram, tokenized_corpus,  V):
    return (tokenized_corpus.count(unigram) + 1)/(len(tokenized_corpus) + V)

def createVocabulary(list_of_tokens):

    Unset_Vocabulary = []
    for list in list_of_tokens:
        for token in list:
            Unset_Vocabulary.append(token)
    V = set(Unset_Vocabulary)
    f = open(r'output_files\Vocabulary', 'w')
    for word in V:
        f.write(str(word) + ",")

def email_process_and_tokenize(text):

    stopwords = nltk.corpus.stopwords.words('english')
    stemmer = SnowballStemmer("english")

    text_rm_p = remove_punc(text)
    tokens = [word.lower() for sent in nltk.sent_tokenize(text_rm_p) for word in nltk.word_tokenize(sent)]
    tokens_rm_sw = [word for word in tokens if word not in stopwords]
    filtered_tokens = []
    for token in tokens_rm_sw:
        if re.search('[a-z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

################################################### MAIN SCRIPT ###################################################

# Term Freq *******************************
# for list in final_tokenized_email_list:
#     count = {}
#     for word in list:
#         try:
#             count[word] += 1
#         except KeyError:
#             count[word] = 1
#     print(count.values())


final_tokenized_email_list =[]

for mail in email_body:
    final_tokenized_email_list.append(email_process_and_tokenize(mail))




# createVocabulary(final_tokenized_email_list)

# unigrams_probs = [0.0] * len(V)
#
# i = 0
# for unigram in V:
#     if unigram != "qwerty":
#         unigrams_probs[i] = calculateUnigramProbLS(unigram, tokenized_corpus, len(V) - 1)
#     i = i + 1











