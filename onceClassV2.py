print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm
train_size =100
test_size =20
outliers_size = 10

xx, yy = np.meshgrid(np.linspace(-8,8,500), np.linspace(-8,8, 500))
classGamma =.1
classNu = 0.1
stepThru = True    # set to TRUE for interactive mode
if (True ):
    frontier_offset =0
    X = .7* np.random.randn(train_size, 2)
    xxx = .7* np.random.normal(.5,3, 200)
    yyy = .7* np.random.normal(.5,.5,200)
    X=np.column_stack((xxx,yyy))
    X_train = X
    X = 1.5 * np.random.randn(test_size, 2)
    X_test = np.r_[X]
    X_outliers = np.random.uniform(low=-4, high=4, size=(outliers_size, 2))
else:
    frontier_offset =3
    #train normal business day
    xxx = .7 * np.random.normal(.5, 3, 200)
    yyy = .7 * np.random.normal(.5, .5, 200)
    X = np.column_stack((xxx, yyy))
    #train normal holiday
    xxx = .7 * np.random.normal(.5, 3, 200)
    yyy = .7 * np.random.normal(.5, .2, 200)
    X2 = np.column_stack((xxx, yyy))
    #concat
    X_train = np.r_[X + 3, X2- 3]
    #sample data
    xxx = .4 * np.random.normal(1 ,2, 20)
    yyy = -.7* np.random.normal(1, .3, 20)
    Y = np.column_stack((xxx, yyy))
    X_test = np.r_[Y + 3, Y - 3]


# fit the model
print("-> Validator starting  (using Non-linear SVM Novelty classfier)")
clf = svm.OneClassSVM(nu=classNu, kernel="rbf", gamma=classGamma)
clf.fit(X_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
n_error_train = y_pred_train[y_pred_train == -1].size
n_error_test = y_pred_test[y_pred_test == -1].size
if (stepThru):

    s = 40
    print("--> Parsing training data ")
    # plot the line, the points, and the nearest vectors to the plane
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.title("File integrity check using a Non-linear SVM Novelty classfier")
    #----------------------1
    plt.axis('tight')
    plt.xlim((-8, 8))
    plt.ylim((-8, 8))
    b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=s, edgecolors='k')
    plt.show()
    #-------------2
    print("----> Creating Frontier")
    plt.axis('tight')
    plt.xlim((-8, 8))
    plt.ylim((-8, 8))
    a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0,2), cmap=plt.cm.PuBu)
    plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')
    b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=s, edgecolors='k')

    plt.show()
    print("-------> Ready to evaluate ")
    #3
    while (True):
        y=[]
        xxx = .7 * np.random.normal(.5, .5, 1)
        yyy = .7 * np.random.normal(.5, 2, 1)

        Y = np.column_stack((xxx, yyy))
        print ("stack:",Y, end= " ----> ")
        X_test = np.r_[Y+frontier_offset]
        y_pred_test = clf.predict(X_test)

        if (y_pred_test== 1):
                print(" File integrity within acceptable tolerance")
        else:
                print ("ATTENTION: todays file is outside allowed tolerance.  investigation required")
        plt.axis('tight')
        plt.xlim((-8, 8))
        plt.ylim((-8, 8))
        a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
        plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')
        b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=s, edgecolors='k')
        b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='blueviolet', s=s, edgecolors='k')
        plt.show()


    #c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='gold', s=s,edgecolors='k')
    plt.axis('tight')
    plt.xlim((-8, 8))
    plt.ylim((-8, 8))
    plt.legend([a.collections[0], b1, b2],
               ["learned frontier", "training files",
                " test files"],
               loc="upper left",
               prop=matplotlib.font_manager.FontProperties(size=11))
    plt.xlabel(
        "ignored in train: %d ; novel files identified: %d ; "

        % (n_error_train, n_error_test))
    plt.show()

else:
    s = 40

    # plot the line, the points, and the nearest vectors to the plane
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.title("File integrity check using a Non-linear SVM Novelty classfier")

    #plt.show()
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0,2), cmap=plt.cm.PuBu)
    a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
    plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')
    b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=s, edgecolors='k')
    b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='blueviolet', s=s, edgecolors='k')
    #plt.show()


    #c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='gold', s=s,edgecolors='k')
    plt.axis('tight')
    plt.xlim((-8, 8))
    plt.ylim((-8, 8))
    plt.legend([a.collections[0], b1, b2],
               ["learned frontier", "training files",
                " test files"],
               loc="upper left",
               prop=matplotlib.font_manager.FontProperties(size=11))
    plt.xlabel(
        "ignored in train: %d ; novel files identified: %d ; "

        % (n_error_train, n_error_test))
    plt.show()


