from __future__ import division
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os

def load_data():
    train = np.genfromtxt('quasar_train.csv', skip_header=True, delimiter=',')
    test = np.genfromtxt('quasar_test.csv', skip_header=True, delimiter=',')
    wavelengths = np.genfromtxt('quasar_train.csv', skip_header=False, delimiter=',')[0]
    return train, test, wavelengths

def add_intercept(X):
    X =np.c_[np.ones(len(X)),X]
    return X

def smooth_data(raw, wavelengths, tau):
    smooth=np.zeros(np.shape(raw))
    for i in range(len(raw)):
        smooth[i] = LWR_smooth(raw[i], wavelengths,tau)
    return smooth

def LWR_smooth(spectrum, wavelengths, tau):
    X = add_intercept(wavelengths)
    theta=np.zeros((len(wavelengths),2))

    # calculating weight and theta in every element of wavelengths
    for i in range(len(wavelengths)):
        diag_el=np.exp(-(wavelengths-wavelengths[i])**2/(2*tau**2))
        w=np.diag(diag_el)
        theta[i]=np.transpose(np.linalg.inv(np.transpose(X) @ w @ X) @ np.transpose(X) @ w @ spectrum)
    smooth_spectrum=theta[:,0]+theta[:,1]*wavelengths
    return smooth_spectrum

def LR_smooth(Y, X_):
    X = add_intercept(X_)
    yhat = np.zeros(Y.shape)
    theta = np.zeros(2)
    theta=np.linalg.inv(np.transpose(X) @ X) @ np.transpose(X) @ Y
    yhat=X @ theta
    return yhat, theta

def plot_b(X, raw_Y, Ys, desc, filename):
    plt.figure()
    plt.plot(X,raw_Y,'ro')
    for i in range(len(Ys)):
        plt.plot(X,Ys[i],label=desc[i])
    plt.xlim(1100,1600)
    plt.legend(loc="best", prop=dict(size=12))
    plt.grid(True)
    plt.xlabel("Wavelength", fontsize=12)
    plt.ylabel("Spectrum f", fontsize=12)
    plt.show()
    plt.savefig(filename)

def plot_c(Yhat, Y, X, filename):
    plt.figure()
    plt.plot(X,Y,'ro',label='smooth test')
    plt.plot(X[:len(Yhat)],Yhat,'b-',label='pred')
    plt.xlim(1100,1600)
    plt.legend(loc="best", prop=dict(size=12))
    plt.grid(True)
    plt.xlabel("Wavelength", fontsize=12)
    plt.ylabel("Spectrum f", fontsize=12)
    plt.show()
    plt.savefig(filename)
    return

def split(full, wavelengths):
    left,right=[],[]
    for i in range(np.shape(full)[0]):
        for j in range(np.shape(full)[1]):
            if wavelengths[j]<1200:
                left.append(full[i,j])
            if wavelengths[j]>=1300:
                right.append(full[i,j])
    left=np.reshape(left,(np.shape(full)[0],-1))
    right=np.reshape(right,(np.shape(full)[0],-1))
    return left, right

def dist(a, b):
    dist = 0
    for i in range(len(a)):
        dist+=(a[i]-b[i])**2
    return dist

def func_reg(left_train, right_train, feat_pred):
    m, n = left_train.shape
    lefthat=np.zeros(np.shape(left_train)[1])
    k=3 # nearest neighbors

    dis=np.zeros(np.shape(right_train)[0])
    for j in range(np.shape(right_train)[0]): # loop through training data (200x300)
        dis[j]=dist(right_train[j,:],feat_pred[:])
    ind=np.argsort(dis)[0:k]
    h=np.max(dis)
    upval=[]
    for w in range(k): # loop through nearest neighbors
        upval.append(ker(dis[ind[w]]/h))
    for y in range(np.shape(left_train)[1]):
        lefthat[y]=np.sum(upval*left_train[ind,y])/sum(upval)
    return lefthat

def ker(x):
    return np.max([1-x,0])

def main():

    raw_train, raw_test, wavelengths = load_data()

    ## Part b.i
    lr_est, theta = LR_smooth(raw_train[0], wavelengths)
    print('Part b.i) Theta=[%.4f, %.4f]' % (theta[0], theta[1]))
    plot_b(wavelengths, raw_train[0], [lr_est], ['Regression line'], 'ps1q5b1.png')

    ## Part b.ii
    lwr_est_5 = LWR_smooth(raw_train[0], wavelengths, 5)
    plot_b(wavelengths, raw_train[0], [lwr_est_5], ['tau = 5'], 'ps1q5b2.png')

    ### Part b.iii
    lwr_est_1 = LWR_smooth(raw_train[0], wavelengths, 1)
    lwr_est_10 = LWR_smooth(raw_train[0], wavelengths, 10)
    lwr_est_100 = LWR_smooth(raw_train[0], wavelengths, 100)
    lwr_est_1000 = LWR_smooth(raw_train[0], wavelengths, 1000)
    plot_b(wavelengths, raw_train[0],
             [lwr_est_1, lwr_est_5, lwr_est_10, lwr_est_100, lwr_est_1000],
             ['tau = 1', 'tau = 5', 'tau = 10', 'tau = 100', 'tau = 1000'],
             'ps1q5b3.png')

    ### Part c.i
    smooth_train, smooth_test = [smooth_data(raw, wavelengths, 5) for raw in [raw_train, raw_test]]

    #### Part c.ii
    left_train, right_train = split(smooth_train, wavelengths)
    left_test, right_test = split(smooth_test, wavelengths)

    train_errors = [dist(left, func_reg(left_train, right_train, right)) for left, right in zip(left_train, right_train)]
    print('Part c.ii) Training error: %.4f' % np.mean(train_errors))

    ### Part c.iii
    test_errors = [dist(left, func_reg(left_train, right_train, right)) for left, right in zip(left_test, right_test)]
    print('Part c.iii) Test error: %.4f' % np.mean(test_errors))

    left_1 = func_reg(left_train, right_train, right_test[0])
    plot_c(left_1, smooth_test[0], wavelengths, 'ps1q5c3_1.png')
    left_6 = func_reg(left_train, right_train, right_test[5])
    plot_c(left_6, smooth_test[5], wavelengths, 'ps1q5c3_6.png')
    pass

if __name__ == '__main__':
    main()
