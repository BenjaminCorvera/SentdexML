### with k nearest neighbors, k should be odd and greater than the number of categories/groups so there are no ties.
# To find distance of points, we use simple euclidian math.
# The larger the dataset, the worse this algorithm is (scaling is bad). SVM's are much better at classification. ###
import pandas as pd
from sklearn import preprocessing, model_selection, neighbors, svm
import numpy as np

df = pd.read_csv("Classification/breast-cancer-wisconsin.data")
df.replace("?", -99999, inplace=True)
df.drop(["id"], axis=1, inplace=True)

X = np.array(df.drop(["class"], axis=1))
X = preprocessing.scale(X)
y = np.array(df["class"])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = svm.SVC()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

# this is where you would pickle the classifier

example_measures = np.array(
    [
        [4, 2, 1, 1, 1, 2, 3, 2, 1],
        [4, 2, 1, 1, 2, 2, 3, 2, 1],
        [4, 8, 1, 1, 2, 2, 3, 2, 1],
    ]
)
# example_measures = np.reshape(len(example_measures),-1)

prediction = clf.predict(example_measures)
print(prediction)
