import random as rd
import scipy.spatial as spatial

def euc(a,b):#a is training data while b is testing data
	return spatial.distance(features)

class ScrappyKNN(object):
	"""docstring for ScrappyKNN"""
	def fit(self,x_train,y_train):
		self.x_train = x_train
		self.y_train = y_train

	def predict(self,x_test):
		predictions = []
		for x in x_test:
			predictions.append(rd.choice(self.y_train))
		return predictions

import sklearn.datasets as datasets
iris = datasets.load_iris()
x = iris.data
y = iris.target

import sklearn.cross_validation as cross_validation #import train_test_split as tts
x_trian,x_test, y_train, y_test = cross_validation.train_test_split(x,y,test_size = .5)

#下面两行可以替换成其他的分类器
# import sklearn.neighbors as neighbors	#KNearestNeighbors
# my_clf = neighbors.KNeighborsClassifier()
my_clf = ScrappyKNN()

my_clf.fit(x_trian,y_train)
predictions = my_clf.predict(x_test)

import sklearn.metrics as metrics
print(metrics.accuracy_score(y_test,predictions))