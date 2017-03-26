import nltk, math , re, time, os, codecs, email.parser
from nltk.stem.snowball import SnowballStemmer
from tools import remove_punc
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections import Counter
import numpy as np


ham_dir = "C:\\Corpus\\CSDMC2010_SPAM\\TRAINING_HAM"
ham_test = "C:\Corpus\CSDMC2010_SPAM\SMALL_TRAINNING_HAM"
spam_dir = "C:\\Corpus\\CSDMC2010_SPAM\\TRAINING_SPAM"

tokenize = lambda doc: doc.lower().split(" ")

def extractBody(filename):
    if not os.path.exists(filename): # dest path doesnot exist
        print ("ERROR: input file does not exist:", filename)
        os._exit(1)
    fp = codecs.open(filename, mode='r', encoding='utf-8', errors='ignore')
    msg = email.message_from_file(fp)
    payload = msg.get_payload()
    if type(payload) == type(list()) :
        payload = payload[0] # only use the first part of payload
    sub = msg.get('subject')
    sub = str(sub)
    if type(payload) != type('') :
        payload = str(payload)
    return payload

def pre_process(text):
    text_rm_p = remove_punc(text)
    return text_rm_p

def word_tokenize(text):
    stopwords = nltk.corpus.stopwords.words('english')
    stemmer = SnowballStemmer("english")
    tokens = [word.lower() for sent in nltk.sent_tokenize(pre_process(text)) for word in nltk.word_tokenize(sent)]
    tokens_rm_sw = [word for word in tokens if word not in stopwords]
    filtered_tokens = []
    for token in tokens_rm_sw:
        if re.search('[a-z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

def inv_doc_freq(tokenized_documents):
    idf_values = {}
    all_tokens_set = set([item for sublist in tokenized_documents for item in sublist])
    for tkn in all_tokens_set:
        contains_token = map(lambda doc: tkn in doc, tokenized_documents)
        idf_values[tkn] = 1 + math.log(len(tokenized_documents)/(sum(contains_token)))
    return idf_values

def term_freq(term, tokenized_document):
    count = tokenized_document.count(term)
    if count == 0:
        return 0
    return 1 + count

def tf_idf(list_of_docs):
    tokenized_documents = []
    for doc in list_of_docs:
        tokenized_documents.append(tokenize(doc))
    idf = inv_doc_freq(tokenized_documents)
    tfidf_documents = []
    for document in tokenized_documents:
        doc_tfidf = []
        for term in idf.keys():
            tf = term_freq(term, document)
            doc_tfidf.append(tf*idf[term])
        tfidf_documents.append(doc_tfidf)
    return tfidf_documents

########################################## MAIN ##########################################
start_time = time.time()
print("--- %s seconds ---" % (time.time() - start_time), "\n")


list_of_docs = []
tokenized_docs = []
files = os.listdir(ham_dir)

'''CREATE LIST OF DOCUMENTS'''
for file in files:
    srcpath = os.path.join(ham_dir, file)
    body_text = extractBody(srcpath)
    list_of_docs.append(body_text)


# '''CREATE LIST OF LISTS OF TOKENS'''
# for file in files:
#     srcpath = os.path.join(ham_test, file)
#     body_text = extractBody(srcpath)
#     tokenized_docs.append(word_tokenize(body_text))


######################################################################################
'''TERM FREQUENCY WIT COUNTVECTORIZER'''

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


tfidf_results = tf_idf(list_of_docs)
print(tfidf_results)

print('############## TIME ################')
print("--- %s seconds ---" % (time.time() - start_time))
print('###############################'+'\n')