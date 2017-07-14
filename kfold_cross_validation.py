import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

#using iris data set
df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/bezdekIris.data",
names = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width", "Class"])

models = []
models.append(("LoR", LogisticRegression()) )
models.append(("LDA", LinearDiscriminantAnalysis()) )
models.append(("QDA", QuadraticDiscriminantAnalysis()) )
models.append(("NB", GaussianNB() ))
models.append(("KNN", KNeighborsClassifier()) )
models.append(("SVM", SVC()) )
models.append(("DT", DecisionTreeClassifier()) )
models.append(("RF", RandomForestClassifier()) )

model_names = []
means = []
stds = []

#shuffle our data and we use 121 out of 150 as training data
data_array = df.values
np.random.shuffle(data_array)
X_learning = data_array[:121][:,0:4]
Y_learning = data_array[:121][:,4]

#split our data in 10 folds
kfold = model_selection.KFold(n_splits=10)

def showSplitting(X_learning):
    for train_index, test_index in kfold.split(X_learning):
        print("Train Index:")
        print(train_index)
        print("Test Index:")
        print(test_index)  

#uncomment following to x how do samples split        
#showSplitting(X_learning)        

for name, model in models:
     #cross validation among models, score based on accuracy
     cv_results = model_selection.cross_val_score(model, X_learning, Y_learning, scoring='accuracy', cv=kfold )
     print("\n"+name)
     model_names.append(name)
     print("Result: "+str(cv_results))
     print("Mean: " + str(cv_results.mean()))
     print("Standard Deviation: " + str(cv_results.std()))
     means.append(cv_results.mean())
     stds.append(cv_results.std())
        
#plot the graphs with bar chart
x_loc = np.arange(len(models)) # the x locations for the groups
width = 0.5   # bar width

models_graph = plt.bar(x_loc, means, width, yerr=stds)
plt.ylabel('Accuracy')
plt.title('Scores by models')
plt.xticks(x_loc, model_names) # models name on x-axis

#add valve on the top of every bar
def addLabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%f' % height, ha='center', 
                 va='bottom')

addLabel(models_graph)

#with bigger graph size
plt.figure(figsize=(150,150)) 
plt.show()
