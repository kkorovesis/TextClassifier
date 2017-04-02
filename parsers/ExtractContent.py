#!/usr/bin/python
# FileName: Subsampling.py 
# Version 1.0 by Tao Ban, 2010.5.26
# This function extract all the contents, ie subject and first part from the .eml file 
# and store it in a new file with the same name in the dst dir. 

import email.parser 
import os, sys, stat, codecs
from os import _exit
import shutil

trainsrcdir="C:\\Corpus\\CSDMC2010_SPAM\\TRAINING"
trainhamdstdir = "C:\\Corpus\\CSDMC2010_SPAM\\TRAINING_HAM"
trainspamdstdir = "C:\\Corpus\\CSDMC2010_SPAM\\TRAINING_SPAM"

testsrcdir="C:\\Corpus\\CSDMC2010_SPAM\\TESTING"
testhamdstdir = "C:\\Corpus\\CSDMC2010_SPAM\\TESTING_HAM"
testspamdstdir = "C:\\Corpus\\CSDMC2010_SPAM\\TESTING_SPAM"

label_dir = "C:\\Corpus\\CSDMC2010_SPAM\\SPAMTrain.label"

def ExtractSubPayload (filename):
    ''' Extract the subject and payload from the .eml file.

    '''
    if not os.path.exists(filename): # dest path doesnot exist
        print ("ERROR: input file does not exist:", filename)
        _exit(1)
    fp = codecs.open(filename, mode='r', encoding='utf-8', errors='ignore')
    msg = email.message_from_file(fp)
    payload = msg.get_payload()
    if type(payload) == type(list()) :
        payload = payload[0] # only use the first part of payload
    sub = msg.get('subject')
    sub = str(sub)
    if type(payload) != type('') :
        payload = str(payload)

    return sub + payload

def ExtractBodyFromDir(srcdir, hamstpath, spamdstpath, dictlabel):
    spam_counter = 0
    ham_counter = 0

    '''1 stands for a HAM and 0 stands for a SPAM'''

    files = os.listdir(srcdir)
    for file in files:
        srcpath = os.path.join(srcdir, file)
        filelabel = search(dict_label,file)
        if filelabel == 1:
            ham_counter +=1
            body = ExtractSubPayload(srcpath)
            dstpath = os.path.join(hamstpath, file)
            dstfile = open(dstpath, 'w', encoding='utf-8')
            dstfile.write(body)
            dstfile.close()
        elif filelabel == 0:
            spam_counter +=1
            body = ExtractSubPayload(srcpath)
            dstpath = os.path.join(spamdstpath, file)
            dstfile = open(dstpath, 'w', encoding='utf-8')
            dstfile.write(body)
            dstfile.close()
    print("Found:",ham_counter," Ham mails"," and ",spam_counter, " Spam mails")

def search(diction, name):
    for k, v in diction.items():
        if name == k:
            return v
    return None

###################################################################
# main function start here

dict_label = {}
with open(label_dir) as f:
    for line in f:
       (val, key) = line.split()
       dict_label[key] = int(val)

ExtractBodyFromDir ( trainsrcdir, trainhamdstdir, trainspamdstdir, dict_label)

ExtractBodyFromDir ( testsrcdir, testhamdstdir, testspamdstdir, dict_label)

# Found: 2224  Ham mails  and  1022  Spam mails
# Found: 725  Ham mails  and  356  Spam mails