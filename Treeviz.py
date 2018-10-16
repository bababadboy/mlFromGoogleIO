import sklearn.tree as tree
import sklearn.datasets as datasets
import numpy as np

iris = datasets.load_iris()
test_idx = [0,50,100]
#training datasets
train_data = np.delete(iris.data, test_idx, axis=0)
train_target = np.delete(iris.target, test_idx)

#test data
test_data, test_target = iris.data[test_idx], iris.target[test_idx]
 
#生成一个空的分类器
clf = tree.DecisionTreeClassifier()
#根据特征和对应的训练结果构造分类器
clf = clf.fit(train_data, train_target)
#使用分类器进行预测
predict = clf.predict(test_data)
print("predict:",predict)
#预测的正确率
import sklearn.metrics as metrics
print("accuracy: ",metrics.accuracy_score(test_target,predict))
#可视化决策树
import pydotplus
dot_data= tree.export_graphviz(clf,
						out_file=None,
						feature_names=iris.feature_names,
						class_names=iris.target_names,
						filled=True,rounded=True,impurity=False)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf('IrisTree.pdf')
# for  g in graph:
# 	g.write_pdf("IrisTree.pdf")  


