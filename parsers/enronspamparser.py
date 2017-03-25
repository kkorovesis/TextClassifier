import os
from email.parser import Parser

spam_directory = "C:\Corpus\CSDMC2010_SPAM\TRAINING_SPAM"
# spam_rootdir = "C:\\Corpus\\Enron Spam"

spam_rootdir = ""
def spam_email_lebelling(email_labels,label):
    email_labels.append(label)

# Email Analuzer
def email_analyse(inputfile, to_email_list, from_email_list, spam_email_body):
    with open(inputfile, "r") as f:
        data = f.read()

    email = Parser().parsestr(data)
    to_email_list.append(email['to'])
    from_email_list.append(email['from'])
    subject_email_list.append(email['subject'])
    spam_email_body.append(email.get_payload())

to_email_list = []
from_email_list = []
subject_email_list = []
spam_email_body = []
spam_email_label = []

# Get All ham mails in directory, Ham Label = 0
for directory, subdirectory, filenames in  os.walk(spam_rootdir):
    for filename in filenames:
        email_analyse(os.path.join(directory, filename), to_email_list, from_email_list, spam_email_body)
        spam_email_lebelling(spam_email_label,1)

print(spam_email_body)