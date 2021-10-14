from more_itertools.more import one
import pandas as pd
import quandl as Quandl
import math, os, datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
from dotenv import load_dotenv
from matplotlib import style
import pickle

load_dotenv()
style.use("ggplot")

apiKey = os.getenv("QUANDL_API_KEY")

df = Quandl.get("WIKI/GOOGL", api_key=apiKey)
df = df[["Adj. Open", "Adj. High", "Adj. Low", "Adj. Close", "Adj. Volume"]]
df["HL_PCT"] = (df["Adj. High"] - df["Adj. Close"]) / df["Adj. Close"] * 100
df["PCT_change"] = (df["Adj. Close"] - df["Adj. Open"]) / df["Adj. Open"] * 100

df = df[["Adj. Close", "HL_PCT", "PCT_change", "Adj. Volume"]]

forecast_col = "Adj. Close"
# In ml, you can't work with NaN or null data
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.1 * len(df)))
df["label"] = df[forecast_col].shift(-forecast_out)

# features
X = np.array(df.drop(["label"], 1))
X = preprocessing.scale(X)
# stuff we are going to predict against (we don't have a y value for these)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)
# labels
y = np.array(df["label"])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = LinearRegression(n_jobs=-1)
# fit is synonoumous with train
clf.fit(X_train, y_train)

# Pickling, saving classifier so training does not occur every time script is run
# with open("linear_regression.pickle", "wb") as f:
#     pickle.dump(clf, f)

# pickle_in = open("linear_regression.pickle", "rb")
# clf = pickle.load(pickle_in)

# score is synonomous with test
accuracy = clf.score(X_test, y_test)

# this is the most important part. This can take a single value or an array
forecast_set = clf.predict(X_lately)

print(forecast_set, accuracy, forecast_out)

df["Forecast"] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

df["Adj. Close"].plot()
df["Forecast"].plot()
plt.legend(loc=4)
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()
