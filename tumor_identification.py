import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import timeit as t

dataset = pd.read_csv('tumor_classification.csv')

X = dataset.iloc[:, 0:30]

y = dataset.select_dtypes(include=[object])

start = t.default_timer()

le = preprocessing.LabelEncoder()
y = y.apply(le.fit_transform)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter=569)
mlp.fit(X_train, y_train.values.ravel())

predictions = mlp.predict(X_test)

stop = t.default_timer()
time_elapsed = stop - start

print(time_elapsed)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))