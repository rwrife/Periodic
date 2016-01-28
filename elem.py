import random
import pandas
import numpy as np
import tensorflow as tf
import skflow
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import datasets, metrics
from sklearn import preprocessing
from sklearn.metrics import classification_report

#15	Phosphorus	P	30.973762	3	15	solid	cub	Nonmetal	0.38	1.2	2.19	10.4867	1.82	317.25	553	7	BranBrand	1669	0.769	[Ne] 3s2 3p3	3	15
P = pandas.DataFrame(columns=['Atomic Weight','Period','Group','Ionic Radius','Density','Melting Point (K)','Boiling Point (K)','Isotopes'])
P.loc[1] = [30.973762,3,15,0.380,1.820000, 317.250, 553.00, 7]

data = pandas.read_csv('data.csv').fillna(0)

y, x = data['Atomic Number'], data[['Atomic Weight','Period','Group','Ionic Radius','Density','Melting Point (K)','Boiling Point (K)','Isotopes']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

lr = LogisticRegression()
lr.fit(x_train, y_train)
print accuracy_score(lr.predict(x_test), y_test)

random.seed(42)
classifier = skflow.TensorFlowDNNClassifier(hidden_units=[10, 20, 10], n_classes=200)
classifier.fit(x_train, y_train)
print accuracy_score(classifier.predict(x_test), y_test)
print classifier.predict(P)
