import os, codecs
import email.parser

ham_dir = "C:\\Corpus\\CSDMC2010_SPAM\\TRAINING_HAM"
spam_dir = "C:\\Corpus\\CSDMC2010_SPAM\\TRAINING_SPAM"

def writeBody(bodylist,dir):

    f = open(dir, 'w',  encoding='utf-8')
    for body in bodylist:
        f.write(body + ",")

def ExtractBody(filename,body):

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

###################################### MAIN ######################################

email_body = []
spam_email_body = []
email_label = []

files = os.listdir(ham_dir)
for file in files:
    srcpath = os.path.join(ham_dir, file)
    email_body.append(ExtractBody(srcpath,email_body))
    email_label.append(1)

files = os.listdir(spam_dir)
for file in files:
    srcpath = os.path.join(spam_dir, file)
    spam_email_body.append(ExtractBody(srcpath,spam_email_body))
    email_label.append(0)


