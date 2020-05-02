import numpy as np
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(123)
tau = 8.

def readMatrix(file):
    fd = open(file, 'r')
    hdr = fd.readline()
    rows, cols = [int(s) for s in fd.readline().strip().split()]
    tokens = fd.readline().strip().split()
    matrix = np.zeros((rows, cols))
    Y = []
    for i, line in enumerate(fd):
        nums = [int(x) for x in line.strip().split()]
        Y.append(nums[0])
        kv = np.array(nums[1:])
        k = np.cumsum(kv[:-1:2])
        v = kv[1::2]
        matrix[i, k] = v
    category = (np.array(Y) * 2) - 1
    return matrix, tokens, category

def svm_train(matrix, category):
    state = {}
    M, N = matrix.shape
    #####################
    Y = category
    matrix = 1. * (matrix > 0)
    squared = np.sum(matrix * matrix, axis=1)
    gram = matrix.dot(matrix.T)
    K = np.exp(-(squared.reshape((1, -1)) + squared.reshape((-1, 1)) - 2 * gram) / (2 * (tau ** 2)) )

    alpha = np.zeros(M)
    alpha_avg = np.zeros(M)
    L = 1. / (64 * M)
    outer_loops = 40

    alpha_avg
    for ii in range(outer_loops * M):
        i = int(np.random.rand() * M)
        margin = Y[i] * np.dot(K[i, :], alpha)
        grad = M * L * K[:, i] * alpha[i]
        if (margin < 1):
            grad -=  Y[i] * K[:, i]
        alpha -=  grad / np.sqrt(ii + 1)
        alpha_avg += alpha

    alpha_avg /= (ii + 1) * M

    state['alpha'] = alpha
    state['alpha_avg'] = alpha_avg
    state['Xtrain'] = matrix
    state['Sqtrain'] = squared
    ####################
    return state

def svm_test(matrix, state):
    M, N = matrix.shape
    output = np.zeros(M)
    ###################
    Xtrain = state['Xtrain']
    Sqtrain = state['Sqtrain']
    matrix = 1. * (matrix > 0)
    squared = np.sum(matrix * matrix, axis=1)
    gram = matrix.dot(Xtrain.T)
    K = np.exp(-(squared.reshape((-1, 1)) + Sqtrain.reshape((1, -1)) - 2 * gram) / (2 * (tau ** 2)))
    alpha_avg = state['alpha_avg']
    preds = K.dot(alpha_avg)
    output = np.sign(preds)
    ###################
    return output

def evaluate(output, label):
    error = (output != label).sum() * 1. / len(output)
    return error

def train_size(no_train):
    error=np.zeros(len(no_train))
    testMatrix, tokenlist, testCategory = readMatrix('MATRIX.TEST')
    for i in range(len(no_train)):
        filename='MATRIX.TRAIN.'+str(no_train[i])
        trainMatrix, tokenlist, trainCategory = readMatrix(filename)
        state = svm_train(trainMatrix, trainCategory)
        output = svm_test(testMatrix, state)
        error[i]=evaluate(output, testCategory)
        print('Test Data Error using %d Data: %1.9f' % (no_train[i],error[i]))
    return error

def error_plot(x,y,filename):
    plt.figure()
    plt.plot(x,y,'ro')
    plt.xlabel('Training size')
    plt.ylabel('Error')
    plt.title('SVM')
    plt.grid(True)
    plt.show()
    plt.savefig(filename)
    return

def main():
    os.chdir('C:\\Users\\E460\\PycharmProjects\\untitled3\\CS229\\hw2\\no6')
    trainMatrix, tokenlist, trainCategory = readMatrix('MATRIX.TRAIN.400')
    testMatrix, tokenlist, testCategory = readMatrix('MATRIX.TEST')

    state = svm_train(trainMatrix, trainCategory)
    output = svm_test(testMatrix, state)

    error=evaluate(output, testCategory)
    print('Full Dataset Error: %1.7f' % error)

    # Answer 6d
    print('\nSVM Training Size Sensitivity')
    no_train=[50,100,200,400,800,1400]
    err_vars=train_size(no_train)
    error_plot(no_train,err_vars,'hw2no6d.png')

    return

if __name__ == '__main__':
    main()
