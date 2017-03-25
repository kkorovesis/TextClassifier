import os, codecs
from email.parser import Parser

# ham_rootdir = "C:\\Corpus\\Enron Ham"
# ham_rootdir = "C:\\Corpus\\Enron Ham\\farmer-d"
# ham_rootdir = "C:\\Corpus\\Enron Ham\\kaminski-v\\azurix_azurix"
ham_rootdir = "C:\Corpus\CSDMC2010_SPAM\TRAINING_NEW"

spam_rootdir = ""
def email_lebelling(email_labels,label):
    email_labels.append(label)

# Email Analuzer
def email_analyse(inputfile, to_email_list, from_email_list, email_body):
    with codecs.open(inputfile, mode='r', encoding='utf-8', errors='ignore') as f:
        data = f.read()

    email = Parser().parsestr(data)
    to_email_list.append(email['to'])
    from_email_list.append(email['from'])
    subject_email_list.append(email['subject'])
    email_body.append(email.get_payload())

to_email_list = []
from_email_list = []
subject_email_list = []
email_body = []
email_label = []


# Get All ham mails in directory, Ham Label = 0
for directory, subdirectory, filenames in  os.walk(ham_rootdir):
    for filename in filenames:
        email_analyse(os.path.join(directory, filename), to_email_list, from_email_list, email_body)
        email_lebelling(email_label,0)
