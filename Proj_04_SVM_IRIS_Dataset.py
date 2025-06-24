import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

irisset = datasets.load_iris()
X=irisset.data[:100,:2]
y=irisset.target[:100]
clf=svm.SVC(kernel='linear')
clf.fit(X, y)
Ypred=clf.predict(X)
w=clf.coef_[0]

xpoints = np.linspace(4,7)
ypoints = -w[0] / w[1] * xpoints - clf.intercept_[0]/ w[1]
plt.plot(xpoints, ypoints, 'g-')
plt.scatter(X[:, 0], X[:, 1], c = y)
plt.suptitle('SVM IRIS Data')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.grid(1,which='both')
plt.axis('tight')
plt.show()


cmat = confusion_matrix(y, Ypred)
print('Confusion matrix of SVC is \n',cmat,'\n')

disp = ConfusionMatrixDisplay(confusion_matrix=cmat)
disp.plot()
plt.show()

SVCscore = accuracy_score(y, Ypred )
print('Accuracy score of SVC is',100*SVCscore,'%\n')