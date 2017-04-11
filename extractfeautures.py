import nltk, math , re, time, os, codecs, email.parser, operator, sys, pickle
from nltk.stem.snowball import SnowballStemmer
from tools import remove_punc
from collections import Counter
import numpy as np
np.set_printoptions(threshold=np.inf)


ham_dir = "C:\\Corpus\\CSDMC2010_SPAM\\TRAINING_HAM"
spam_dir = "C:\\Corpus\\CSDMC2010_SPAM\\TRAINING_SPAM"
test_ham_dir = "C:\\Corpus\\CSDMC2010_SPAM\\TESTING_HAM"
test_spam_dir = "C:\\Corpus\\CSDMC2010_SPAM\\TESTING_SPAM"


ham_small = "C:\\Corpus\\CSDMC2010_SPAM\\SMALL_TRAINNING_HAM"
spam_small = "C:\\Corpus\\CSDMC2010_SPAM\\SMALL_TRAINNING_SPAM"
test_ham_small = "C:\\Corpus\\CSDMC2010_SPAM\\SMALL_TESTING_HAM"
test_spam_small = "C:\\Corpus\\CSDMC2010_SPAM\\SMALL_TESTING_SPAM"

# tokenize = lambda doc: doc.lower().split(" ")

def writeFeaturesDict(features):

    f = open(r'output_files\features_kv', 'w', encoding='utf-8')
    for dict in features:
        for key, value in dict.items():
            f.write( key + value) # doesn't work json needed

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
    # text = pre_process(text) # remove punctuations from text / doesn't work well with sent_tokenize
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    tokens_rm_sw = [word for word in tokens if word not in stopwords]
    filtered_tokens = []
    for token in tokens_rm_sw:
        if re.search('[a-z]', token):
            filtered_tokens.append(token)
    all_stems = [stemmer.stem(t) for t in filtered_tokens]
    stems = remove_rare(all_stems,4)
    return stems

def inv_doc_freq(tokenized_documents): # calculates the inverse document frequency
    idf_values = {}
    all_tokens_set = set([item for sublist in tokenized_documents for item in sublist])
    # all_tokens = [item for sublist in tokenized_documents for item in sublist]
    for token in all_tokens_set:
        contains_token = map(lambda doc: token in doc, tokenized_documents)
        idf_values[token] = 1 + math.log(len(tokenized_documents)/ (sum(contains_token)),2)
    return idf_values

def term_freq(term, tokenized_document): # calculates the term frequency
    count = tokenized_document.count(term)
    if count == 0:
        return 0
    return 1 + count



def tf_idf(list_of_docs): # returns a list of the tf-idf scores of all docs
    tokenized_documents = []
    for doc in list_of_docs:
        tokenized_documents.append(word_tokenize(doc))
    idf = inv_doc_freq(tokenized_documents)
    if not tokenized_documents:
        print("no data to process")
        sys.exit()
    tfidf_documents = []
    names = feature_selection(idf, 5000)
    for document in tokenized_documents:
        doc_tfidf = []
        for term in names:
            tf = term_freq(term, document)
            term_tfidf = float("{0:.2f}".format(tf*idf[term]))
            doc_tfidf.append(term_tfidf)
        tfidf_documents.append(doc_tfidf)
    return tfidf_documents,names,idf

def tf_idf_testing(list_of_docs,idf,names):
    tokenized_documents = []
    for doc in list_of_docs:
        tokenized_documents.append(word_tokenize(doc))
    if not tokenized_documents:
        print("no data to process")
        sys.exit()
    tfidf_documents = []
    for document in tokenized_documents:
        doc_tfidf = []
        for term in names:
            tf = term_freq(term, document)
            term_tfidf = float("{0:.2f}".format(tf * idf[term]))
            doc_tfidf.append(term_tfidf)
        tfidf_documents.append(doc_tfidf)
    return tfidf_documents



def feature_selection(idf,max_feautures):
    sorted_x = sorted(idf.items(), key=operator.itemgetter(1), reverse=True)
    return [y[0] for y in sorted_x][:max_feautures]

def load_data():
    ######################################################################################
    '''CREATE THE TRAIN FEATURES'''
    list_of_ham_docs = []
    ham_labels = []
    list_of_spam_docs = []
    spam_labels = []
    #
    # ham_files = os.listdir(ham_small)
    # spam_files = os.listdir(spam_small)

    ham_files = os.listdir(ham_dir)
    spam_files = os.listdir(spam_dir)

    '''CREATE LIST OF DOCUMENTS'''
    for file in ham_files:
        # srcpath = os.path.join(ham_small, file)
        srcpath = os.path.join(ham_dir, file)
        body_text = extractBody(srcpath)
        list_of_ham_docs.append(body_text)
        ham_labels.append(1)
        number_of_hams = len(list_of_ham_docs)
    print("Number of Hams: ", number_of_hams)

    for file in spam_files:
        # srcpath = os.path.join(spam_small, file)
        srcpath = os.path.join(spam_dir, file)
        body_text = extractBody(srcpath)
        list_of_spam_docs.append(body_text)
        spam_labels.append(0)
        number_of_spams = len(list_of_spam_docs)
    print("Number of Spams: ", number_of_spams)

    list_of_docs = list_of_ham_docs + list_of_spam_docs

    print("Number of Docs: ", len(list_of_docs))
    labels_of_docs = ham_labels + spam_labels
    print("Number of Labels: ", len(labels_of_docs))
    ######################################################################################

    ######################################################################################
    '''CREATE THE TEST FEATURES'''
    test_list_of_ham_docs = []
    test_ham_labels = []
    test_list_of_spam_docs = []
    test_spam_labels = []

    # test_ham_files = os.listdir(test_ham_small)
    # test_spam_files = os.listdir(test_spam_small)

    test_ham_files = os.listdir(test_ham_dir)
    test_spam_files = os.listdir(test_spam_dir)

    '''CREATE LIST OF DOCUMENTS'''
    for file in test_ham_files:
        # test_srcpath = os.path.join(test_ham_small, file)
        test_srcpath = os.path.join(test_ham_dir, file)
        test_body_text = extractBody(test_srcpath)
        test_list_of_ham_docs.append(test_body_text)
        test_ham_labels.append(1)
        test_number_of_hams = len(test_list_of_ham_docs)
    print("Number of Test Hams: ", test_number_of_hams)

    for file in test_spam_files:
        # test_srcpath = os.path.join(test_spam_small, file)
        test_srcpath = os.path.join(test_spam_dir, file)
        test_body_text = extractBody(test_srcpath)
        test_list_of_spam_docs.append(test_body_text)
        test_spam_labels.append(0)
        test_number_of_spams = len(test_list_of_spam_docs)
    print("Number of Test  Spams: ", test_number_of_spams)

    test_list_of_docs = test_list_of_ham_docs + test_list_of_spam_docs
    print("Number of Test Docs: ", len(test_list_of_docs))
    test_labels_of_docs = test_ham_labels + test_spam_labels
    print("Number of Test Labels: ", len(test_labels_of_docs))

    return list_of_docs,test_list_of_docs,labels_of_docs,test_labels_of_docs

