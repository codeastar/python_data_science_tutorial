import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# get csv data file from url, or you can use your file path like 
# read_csv("/Users/yourusername/yourdirectory/yourfilename")
# since there is no header in the csv file, we add one for it with "names"
df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/bezdekIris.data",
names = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width", "Class"])

# display the first 10 data to taste the look and feel
print(df.head(10))
# print the summary of our dataset with mean and distribution info
print(df.describe())

#draw a histogram graph with bins size=20 (larger increaes accuracy but also resource usage) 
df.hist(bins=20)
#display the graph using matplotlib
plt.show()

#data collection 
#randomize the data
data_array = df.values
np.random.shuffle(data_array)

#use 80 data for machine learning, 20 for testing 
X_learning = data_array[:80][:,0:4]
Y_learning = data_array[:80][:,4]

X = data_array[-20:][:,0:4]
Y = data_array[-20:][:,4]

#use support vector machine as learning model
svc = SVC()
svc.fit(X_learning,Y_learning)

predictions = svc.predict(X)

print("Predicted results:")
print(predictions)
print("Actual results:")
print(Y)

print("Accuracy rate:  %f" % (accuracy_score(Y, predictions)))
print(confusion_matrix(Y, predictions))
print(classification_report(Y, predictions))