import nltk, math , re, time, os, codecs, email.parser, itertools
from nltk.stem.snowball import SnowballStemmer
from tools import remove_punc
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections import Counter
import numpy as np
np.set_printoptions(threshold=np.inf)


ham_dir = "C:\\Corpus\\CSDMC2010_SPAM\\TRAINING_HAM"
ham_test = "C:\\Corpus\\CSDMC2010_SPAM\\SMALL_TRAINNING_HAM_ONE"
spam_dir = "C:\\Corpus\\CSDMC2010_SPAM\\TRAINING_SPAM"

# tokenize = lambda doc: doc.lower().split(" ")

def writeAnything(things):

    f = open(r'output_files\things3', 'w', encoding='utf-8')
    for thing in things:
        f.write(str(thing) + "\n")

def writeFeatures(tfidf_results):

    f = open(r'output_files\features', 'w', encoding='utf-8')
    for mail_features in tfidf_results:
        f.write(str(mail_features) + "\n")

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

def remove_rare(tokens,n):
    counter = Counter(tokens)
    red_tokens = [word for word in tokens if counter[word] >= n]
    return red_tokens

def word_tokenize(text):
    stopwords = nltk.corpus.stopwords.words('english')
    stemmer = SnowballStemmer("english")
    tokens = [word.lower() for sent in nltk.sent_tokenize(pre_process(text)) for word in nltk.word_tokenize(sent)]
    tokens_rm_sw = [word for word in tokens if word not in stopwords]
    filtered_tokens = []
    for token in tokens_rm_sw:
        if re.search('[a-z]', token):
            filtered_tokens.append(token)
    all_stems = [stemmer.stem(t) for t in filtered_tokens]
    stems = remove_rare(all_stems,4)
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
        tokenized_documents.append(word_tokenize(doc))
    idf = inv_doc_freq(tokenized_documents)
    tfidf_documents = []
    for document in tokenized_documents:
        doc_tfidf = []
        for term in idf.keys():
            tf = term_freq(term, document)
            term_tfidf = tf*idf[term]
            term_tfidf = float("{0:.2f}".format(term_tfidf))
            doc_tfidf.append(term_tfidf)
        tfidf_documents.append(doc_tfidf)
    return tfidf_documents


########################################## MAIN ##########################################
start_time = time.time()
print("--- %s seconds ---" % (time.time() - start_time), "\n")


list_of_docs = []
tokenized_docs = []
files = os.listdir(ham_test)

'''CREATE LIST OF DOCUMENTS'''
for file in files:
    srcpath = os.path.join(ham_test, file)
    body_text = extractBody(srcpath)
    list_of_docs.append(body_text)

tokenized_documents = []
for doc in list_of_docs:
    tokenized_documents.append(word_tokenize(doc))

idf_list = inv_doc_freq(tokenized_docs)

writeAnything(idf_list)





# '''CREATE LIST OF LISTS OF TOKENS'''
# for file in files:
#     srcpath = os.path.join(ham_test, file)
#     body_text = extractBody(srcpath)
#     tokenized_docs.append(word_tokenize(body_text))


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

tfidf_results = tf_idf(list_of_docs)
# for tfidf in tfidf_results:
#     print("Number of features = ", len(tfidf))

writeAnything(tfidf_results)

list_sum=([float("{0:.2f}".format(sum(x))) for x in zip(*tfidf_results)])

array_sum = np.array(list_sum)

# writeFeatures(tfidf_results)

# print("Number of features in sum= ", len(array_sum))
# print(array_sum)
# print([i[0] for  i in (enumerate(array_sum))])
# print([i[0] for i in sorted(enumerate(list_sum), key=lambda x:x[1])])
# print(sorted(range(len(array_sum)),key=lambda x:array_sum[x]))
# print(sorted((e,i) for i,e in enumerate(array_sum)))
#
# for i,e in enumerate(array_sum):
#     enum_sorted = sorted((e,i),reverse=True)
#
#
# print(enum_sorted)



print('############## TIME ################')
print("--- %s seconds ---" % (time.time() - start_time))
print('###############################'+'\n')