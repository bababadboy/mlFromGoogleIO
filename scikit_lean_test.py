import sklearn.tree as tree
#特征值
features = [[140, 1],[130, 1],[150, 0],[170, 0]]
#结果
labels = [0, 0, 1, 1]
#生成一个空的分类器
clf = tree.DecisionTreeClassifier()
#根据特征和对应的结果构造分类器
clf = clf.fit(features, labels)
#使用分类器进行预测
print(clf)
print(clf.predict([[150,1],[145,1]]))