import numpy as np
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

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
    return matrix, tokens, np.array(Y)

def nb_train(matrix, category):
    state = {}
    m=matrix.shape[0]
    N = matrix.shape[1]

    spam_words=matrix[np.where(category==1),:][0]
    nonspam_words=matrix[np.where(category==0),:][0]

    # Calculating Prior Probabilities
    p_prior_1=np.log(spam_words.shape[0]/m)
    p_prior_0=np.log(nonspam_words.shape[0]/m)

    # Calculating phi & Applying Laplace smoothing
    phi_1=[np.log((np.sum(spam_words[:,i])+1)/(sum(sum(spam_words))+N)) for i in range(matrix.shape[1])]
    phi_0=[np.log((np.sum(nonspam_words[:,i])+1)/(sum(sum(nonspam_words))+N)) for i in range(matrix.shape[1])]

    namelist=['phi_1','phi_0','p_prior_1','p_prior_0']
    varlist=[phi_1,phi_0,p_prior_1,p_prior_0]
    state=dict((namelist[i],varlist[i]) for i in range(len(namelist)))

    return state

def nb_test(matrix, state):
    output = np.zeros(matrix.shape[0])

    for i in range(matrix.shape[0]):
        nonzero_token=np.nonzero(matrix[i,:])

        # Calculating posterior probabilities
        pos_post=np.dot(matrix[i,nonzero_token],np.asarray(state['phi_1'])[nonzero_token])+np.asarray([state['p_prior_1']])
        neg_post=np.dot(matrix[i,nonzero_token],np.asarray(state['phi_0'])[nonzero_token])+np.asarray([state['p_prior_0']])

        output[i]=0 if neg_post>pos_post else 1
    return output

def evaluate(output, label):
    error = (output != label).sum() * 1. / len(output)
    return error

def find_tokens(state,tokenlist):
    tokenrank=np.zeros(len(state['phi_1']))

    # Finding top 5 tokens
    for i in range(len(state['phi_1'])):
        tokenrank[i]=np.log(state['phi_1'][i]/state['phi_0'][i])
    ind=tokenrank.argsort()[:5]

    # Obtaining top 5 tokens from the list
    top_5=[]
    for j in ind:
        top_5.append(tokenlist[j])
    print('Top 5 indicative tokens: %s' % top_5)
    return top_5

def train_size(no_train):
    error=np.zeros(len(no_train))
    for i in range(len(no_train)):
        filename='MATRIX.TRAIN.'+str(no_train[i])
        trainMatrix, tokenlist, trainCategory = readMatrix(filename)
        testMatrix, tokenlist, testCategory = readMatrix('MATRIX.TEST')
        state = nb_train(trainMatrix, trainCategory)
        output = nb_test(testMatrix, state)
        error[i]=evaluate(output, testCategory)
        print('Test Data Error using %d Data: %1.4f' % (no_train[i],error[i]))
    return error

def error_plot(x,y,filename):
    plt.figure()
    plt.plot(x,y,'ro')
    plt.xlabel('Training size')
    plt.ylabel('Error')
    plt.title('Naive Bayes')
    plt.grid(True)
    plt.show()
    plt.savefig(filename)
    return

def main():
    os.chdir('C:\\Users\\E460\\PycharmProjects\\untitled3\\CS229\\hw2\\no6')
    trainMatrix, tokenlist, trainCategory = readMatrix('MATRIX.TRAIN')
    testMatrix, tokenlist, testCategory = readMatrix('MATRIX.TEST')

    state = nb_train(trainMatrix, trainCategory)
    output = nb_test(testMatrix, state)

    # Answer 6a
    error=evaluate(output, testCategory)
    print('6a. Error: %1.4f \n' % error)

    # Answer 6b
    find_tokens(state,tokenlist)

    # Answer 6c
    print('\nNaive Bayes Training Size Sensitivity')
    no_train=[50,100,200,400,800,1400]
    err_vars=train_size(no_train)
    error_plot(no_train,err_vars,'hw2no6c.png')

    # Answer 6d
    print('\n')

    return

if __name__ == '__main__':
    main()
