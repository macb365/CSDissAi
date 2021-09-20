import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC 

dataset = pd.read_csv("traintest-dataset.csv")
dataset.info()

array = dataset.values
X = array[:,0:562]
Y = array[:,562]
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=0.25)
    
models = []
models.append(('LR', LogisticRegression(max_iter=8000)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN-2', KNeighborsClassifier(n_neighbors=2)))
models.append(('KNN-3', KNeighborsClassifier(n_neighbors=3)))
models.append(('DT', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
    
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    


    lr = LogisticRegression(max_iter = 8000)
    lr.fit(X_train, Y_train)
    preds = lr.predict(X_validation)
    print(lr)
    print(accuracy_score(Y_validation, preds))
    print(confusion_matrix(Y_validation, preds))
    print(classification_report(Y_validation, preds))
    
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, Y_train)
    preds = lda.predict(X_validation)
    print(lda)
    print(accuracy_score(Y_validation, preds))
    print(confusion_matrix(Y_validation, preds))
    print(classification_report(Y_validation, preds))
    
    knntwo = KNeighborsClassifier(n_neighbors = 2)
    knntwo.fit(X_train, Y_train)
    preds = knntwo.predict(X_validation)
    print(knntwo)
    print(accuracy_score(Y_validation, preds))
    print(confusion_matrix(Y_validation, preds))
    print(classification_report(Y_validation, preds))
    
    knnthree = KNeighborsClassifier(n_neighbors = 3)
    knnthree.fit(X_train, Y_train)
    preds = knnthree.predict(X_validation)
    print(knnthree)
    print(accuracy_score(Y_validation, preds))
    print(confusion_matrix(Y_validation, preds))
    print(classification_report(Y_validation, preds))
    
    dt = DecisionTreeClassifier()
    dt.fit(X_train, Y_train)
    preds = dt.predict(X_validation)
    print(dt)
    print(accuracy_score(Y_validation, preds))
    print(confusion_matrix(Y_validation, preds))
    print(classification_report(Y_validation, preds))
    
    NB = GaussianNB()
    NB.fit(X_train, Y_train)
    preds = NB.predict(X_validation)
    print(NB)
    print(accuracy_score(Y_validation, preds))
    print(confusion_matrix(Y_validation, preds))
    print(classification_report(Y_validation, preds))
    break
    
    
    
    
    
