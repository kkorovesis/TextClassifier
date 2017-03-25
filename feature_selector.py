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

import time
from collections import Counter

import math
import nltk
import re
from nltk.stem.snowball import SnowballStemmer

from parsers.ListBuilder import email_body, email_label
from parsers.ListBuilder import  spam_email_body
from tools import remove_punc

start_time = time.time()

print("--- %s seconds ---" % (time.time() - start_time))

# def write_unigram_probs(uprobs,V):
#
#     f = open(r'output_files\unigram_probs', 'w', encoding='utf-8')
#     i = 0
#     for u in uprobs:
#         f.write("P(" + (V[i]) + ") = " + str(u[i]) + "\n")
#         i = + 1

def writeVocabulary(V):

    f = open(r'output_files\Vocabulary', 'w', encoding='utf-8')
    for word in V:
        f.write(str(word) + ",")

def writespamVocabulary(sV):

    f = open(r'output_files\SpamVocabulary', 'w', encoding='utf-8')
    for word in sV:
        f.write(str(word) + ",")

def calculateUnigramProbLS(unigram,unset_V,  V):
    return (unset_V.count(unigram) + 1)/(len(unset_V) + V)

def calculateEntropy(probs):
    entropy = 0.0
    for prob in probs:
        entropy -= prob * math.log(prob,2)
    return entropy

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

def remove_rare(tokens, n):
    temp_counter = Counter(tokens)
    ts = [word for word in tokens if temp_counter[word] >= n]
    # ts = ' '.join(ts)
    return ts
################################################### MAIN SCRIPT ###################################################


final_token_list =[]
final_token_spam_list=[]

for mail in email_body:
    final_token_list.append(email_process_and_tokenize(mail))

for mail in spam_email_body:
    final_token_spam_list.append(email_process_and_tokenize(mail))

all_unset_Vocabulary = []
all_unset_spam_Vocabulary = []

for list in final_token_list:
    for token in list:
        all_unset_Vocabulary.append(token)

for list in final_token_spam_list:
    for token in list:
        all_unset_spam_Vocabulary.append(token)

unset_vocabulary = remove_rare(all_unset_Vocabulary,4)
unset_spam_vocabulary = remove_rare(all_unset_spam_Vocabulary,4)

V = sorted(set(unset_vocabulary))
writeVocabulary(V)

sV = sorted(set(unset_spam_vocabulary))
writespamVocabulary(sV)

##########################################################################################
unigram_probs = [0.0] * len(V)
i = 0
for unigram in V:
    unigram_probs[i] = calculateUnigramProbLS(unigram, unset_vocabulary, len(V) - 4)
    i = i + 1
f = open(r'output_files\unigram_probs', 'w', encoding='utf-8')
i = 0
for unigram in V:
    f.write("P(" + (V[i]) + ") = " + str(unigram_probs[i]) + "\n")
    i = i + 1
f.close()
##########################################################################################
spam_unigram_probs = [0.0] * len(sV)
i = 0
for unigram in sV:
    spam_unigram_probs[i] = calculateUnigramProbLS(unigram, unset_spam_vocabulary, len(sV) - 4)
    i = i + 1
f = open(r'output_files\spam_unigram_probs', 'w', encoding='utf-8')
i = 0
for unigram in sV:
    f.write("P(" + (sV[i]) + ") = " + str(spam_unigram_probs[i]) + "\n")
    i = i + 1
f.close()
##########################################################################################

print('############# HAM ##################')
print('Ham entropy : ' + str(calculateEntropy(unigram_probs)))
print('###############################'+'\n')

print('############## SPAM ################')
print('Spam entropy : ' + str(calculateEntropy(spam_unigram_probs)))
print('###############################'+'\n')

print('############## TIME ################')
print("--- %s seconds ---" % (time.time() - start_time))
print('###############################'+'\n')



