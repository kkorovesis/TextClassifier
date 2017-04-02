from sklearn.metrics import precision_score,recall_score,accuracy_score,f1_score,average_precision_score,roc_auc_score

'''1 stands for a HAM and 0 stands for a SPAM'''

def print_metrics(test_labels,predicted_labels,pos_label=0):
    print("Precision: ", precision_score(test_labels,predicted_labels,pos_label=pos_label))

    print("Recall: ", recall_score(test_labels, predicted_labels, pos_label=pos_label))

    print("Accuracy: ", accuracy_score(test_labels,predicted_labels))

    print("F1: ", f1_score(test_labels, predicted_labels, pos_label=pos_label))

    print("Average Precision: ", average_precision_score(test_labels, predicted_labels))

    print("ROC: ", roc_auc_score(test_labels, predicted_labels))

