#!/usr/bin/python
# FileName: Subsampling.py 
# Version 1.0 by Tao Ban, 2010.5.26
# This function extract all the contents, ie subject and first part from the .eml file 
# and store it in a new file with the same name in the dst dir. 

import email.parser 
import os, sys, stat, codecs
import shutil

srcdir="C:\\Corpus\\CSDMC2010_SPAM\\TRAINING"
hamdstdir = "C:\\Corpus\\CSDMC2010_SPAM\\TRAINING_HAM"
spamdstdir = "C:\\Corpus\\CSDMC2010_SPAM\\TRAINING_SPAM"
label_dir = "C:\\Corpus\\CSDMC2010_SPAM\\SPAMTrain.label"

def ExtractSubPayload (filename):
    ''' Extract the subject and payload from the .eml file.

    '''
    if not os.path.exists(filename): # dest path doesnot exist
        print ("ERROR: input file does not exist:", filename)
        os.exit(1)
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
            dstpath = os.path.join(hamdstdir, file)
            dstfile = open(dstpath, 'w', encoding='utf-8')
            dstfile.write(body)
            dstfile.close()
        elif filelabel == 0:
            spam_counter +=1
            body = ExtractSubPayload(srcpath)
            dstpath = os.path.join(spamdstdir, file)
            dstfile = open(dstpath, 'w', encoding='utf-8')
            dstfile.write(body)
            dstfile.close()
    print("Found:",ham_counter," Ham mails"," and ",spam_counter, " Spam mails")

def search(dict, name):
    for key, value in dict.items():
        if name == key:
            return value
    return None

###################################################################
# main function start here
# srcdir is the directory where the .eml are stored

dict_label = {}
with open(label_dir) as f:
    for line in f:
       (val, key) = line.split()
       dict_label[key] = int(val)


###################################################################
ExtractBodyFromDir ( srcdir, hamdstdir, spamdstdir, dict_label)

