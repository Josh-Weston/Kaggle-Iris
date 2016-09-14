# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 19:35:21 2016

@author: joshWeston
"""

from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.externals.six import StringIO
import pydot

iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

#Graph the decision tree that was created
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph[0].write_pdf('iris.pdf') #this is failing for some reason