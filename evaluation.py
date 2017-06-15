from sklearn.metrics import precision_score,recall_score,accuracy_score,f1_score
from sklearn.svm import SVC, LinearSVC
from sklearn.cross_validation import KFold
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def init_sklearn_classifier(classifier_name, cost=100, n_jobs=4):

    classifier_list = {
        "SVM Linear": SVC(kernel='linear', C=cost),
        "SVM Poly": SVC(kernel='poly', C=cost),
        "SVM rbf": SVC(kernel='rbf', C=cost),
        "Linear SVC": LinearSVC(C=cost),
        "k-NN": KNeighborsClassifier(n_neighbors=100, n_jobs=n_jobs),
        "Random Forests": RandomForestClassifier(n_estimators=350, max_features=20, max_leaf_nodes=600, n_jobs=n_jobs),
        "Logistic Regression L1": LogisticRegression(C=cost, penalty='l1', n_jobs=n_jobs),
        "Logistic Regression L2": LogisticRegression(C=cost, penalty='l1', n_jobs=n_jobs),
        "Logistic Regression Stochastic Gradient Descent" : LogisticRegression(C=cost, penalty='l2',
                                                                    class_weight='balanced',max_iter=500, solver='sag'),
        "Decision Trees": DecisionTreeClassifier(min_samples_leaf=250),
        "SGD": SGDClassifier(alpha=.0001, n_iter=50, penalty="elasticnet", n_jobs=n_jobs),
    }
    return classifier_list[classifier_name]

def evaluate_testing(train_set, train_labels ,test_set, test_labels, classifier_name, n_jobs=4,pos_label=0):

    precision, recall, f1, accuracy = 0.0, 0.0, 0.0, 0.0

    classifier = init_sklearn_classifier(classifier_name, n_jobs)
    classifier.fit(train_set, train_labels)
    predicted_labels = classifier.predict(test_set)

    precision += precision_score(test_labels, predicted_labels, pos_label=pos_label)
    recall += recall_score(test_labels, predicted_labels, pos_label=pos_label)
    f1 += f1_score(test_labels, predicted_labels, pos_label=pos_label)
    accuracy += accuracy_score(test_labels, predicted_labels)

    print_metrics(precision, recall, f1, accuracy, pos_label)


def cross_validation(x, y, classifier_name, n_folds=10, n_jobs=4,pos_label=0):

    """ 1 stands for a HAM and 0 stands for a SPAM """

    precision , recall , f1 , accuracy = 0.0 , 0.0 , 0.0 , 0.0

    kf = KFold(len(x), shuffle=True, n_folds=n_folds)
    for train_index, test_index in kf:
        train_set, test_set, train_labels, test_labels = x[train_index], x[test_index], y[train_index], y[test_index]
        classifier = init_sklearn_classifier(classifier_name, n_jobs)
        classifier.fit(train_set, train_labels)
        predicted_labels = classifier.predict(test_set)

        precision += precision_score(test_labels,predicted_labels,pos_label=pos_label)
        recall += recall_score(test_labels, predicted_labels, pos_label=pos_label)
        f1 += f1_score(test_labels, predicted_labels, pos_label=pos_label)
        accuracy += accuracy_score(test_labels, predicted_labels)

    precision /= n_folds
    recall /= n_folds
    f1 /= n_folds
    accuracy /= n_folds

    print_metrics(precision,recall,f1,accuracy,pos_label)

'''1 stands for a HAM and 0 stands for a SPAM'''

def print_metrics(precision,recall,f1_score,accuracy,pos_label):

    if pos_label == 1:
        print("Predicting Hams")
    else:
        print("Predicting Spam")

    print("Precision: ", precision)
    print("Recall: ", recall)
    print("Accuracy: ", accuracy)
    print("F1: ", f1_score)
    print(" ")
