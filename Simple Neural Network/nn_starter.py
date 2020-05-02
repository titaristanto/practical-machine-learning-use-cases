import numpy as np
import matplotlib.pyplot as plt
import os

def readData(images_file, labels_file):
    x = np.loadtxt(images_file, delimiter=',')
    y = np.loadtxt(labels_file, delimiter=',')
    return x, y


def softmax(x):
    """
    Compute softmax function for input. 
    Use tricks from previous assignment to avoid overflow
    """
	### YOUR CODE HERE
    ex=np.exp(x-np.max(x,axis=0,keepdims=True))
    s=ex/np.sum(ex,axis=0,keepdims=True)
	### END YOUR CODE
    return s

def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    """
    s=1/(1+np.exp(-x))
    return s

def forward_prop(x_inp, y_inp, params):
    """
    return hidden layer, output(softmax) layer and loss
    """
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    ### YOUR CODE HERE
    b1_m=b1[:,0:1]+np.zeros((W1.shape[0],x_inp.shape[0]))
    z1=np.dot(W1,np.transpose(x_inp))+b1_m
    a1=sigmoid(z1)
    h=a1

    # Hidden Layer to Output
    b2_m=b2[:,0:1]+np.zeros((W2.shape[0],a1.shape[1]))
    z2=np.dot(W2,a1)+b2_m
    y=softmax(z2)

    # Loss Calculation
    cost=-np.multiply(y_inp.T,np.log(y)).sum()/x_inp.shape[0]

    ### END YOUR CODE
    return h, y, cost

def backward_prop(x,y,a1,labels, params):
    """
    return gradient of parameters
    """
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    ### YOUR CODE HERE
    gradW2=np.dot((y-labels.T),np.transpose(a1))/x.shape[0] #ok

    gradb2=np.sum(y-labels.T,axis=1,keepdims=True)/x.shape[0] #ok

    g_der=np.multiply(a1,(1-a1))
    delta=np.multiply(np.dot(W2.T,(y-labels.T)),g_der)
    gradW1=np.dot(delta,x)/x.shape[0] #ok

    gradb1=np.sum(delta,axis=1,keepdims=True)/x.shape[0] #ok

    ### END YOUR CODE

    grad = {}
    grad['W1'] = gradW1
    grad['W2'] = gradW2
    grad['b1'] = gradb1
    grad['b2'] = gradb2

    return grad

def nn_train(trainData, trainLabels, devData, devLabels):
    (m, n) = trainData.shape
    v=trainLabels.shape[1]
    alpha=0.0001 # Regularization const
    num_hidden = 300
    learning_rate = 5
    B=50 # batch iterations needed per epoch
    bs=int(m/B)

    ### YOUR CODE HERE
    # Input to Hidden Layer
    W1=np.random.rand(num_hidden,n)*1
    b1=np.zeros((num_hidden,bs))*1
    W2=np.random.rand(v,num_hidden)*1
    b2=np.zeros((v,bs))*1
    params={'W1':W1,'W2':W2,'b1':b1,'b2':b2}

    train_loss=[]
    dev_loss=[]
    train_acc=[]
    dev_acc=[]

    for i in range(31): # iterations over epochs
        for j in range(B): # iterations over batches
            x_inp=trainData[j*1000:(j+1)*1000,:]
            y_inp=trainLabels[j*1000:(j+1)*1000,:]

            h,y_b,cost=forward_prop(x_inp,y_inp,params)
            grad=backward_prop(x_inp,y_b,h,y_inp,params)

            # Parameters update
            params['W1']=params['W1']-learning_rate*grad['W1']+alpha*(np.sum(params['W1']*params['W1']))
            params['W2']=params['W2']-learning_rate*grad['W2']+alpha*(np.sum(params['W2']*params['W2']))
            params['b1']=params['b1']-learning_rate*grad['b1']
            params['b2']=params['b2']-learning_rate*grad['b2']

        # Saving Loss and Accuracy per epoch
        h_tot,y_tot,total_loss=forward_prop(trainData,trainLabels,params)
        h_tot_dev,y_tot_dev,total_loss_dev=forward_prop(devData,devLabels,params)

        train_loss.append(total_loss)
        train_acc.append(compute_accuracy(y_tot.T,trainLabels))
        dev_loss.append(total_loss_dev)
        dev_acc.append(compute_accuracy(y_tot_dev.T,devLabels))

        print("Iteration # %d. Training Loss: %f. Accuracy: %f" % (i, train_loss[i], train_acc[i]))
        print("Iteration # %d. Dev Loss: %f. Accuracy: %f" % (i, dev_loss[i], dev_acc[i]))

    ### END YOUR CODE

    return params

def plot(train_data,dev_data,ytitle):
    plt.figure(1)
    plt.plot(np.arange(1,32),train_data,'r-',label='Training Data')
    plt.plot(np.arange(1,32),dev_data,'b-',label='Dev Data')
    plt.xlabel('Epochs')
    plt.ylabel(ytitle)
    plt.ylim([0,1])
    plt.grid(True)
    plt.legend(loc="best", prop=dict(size=12))
    plt.show()
    plt.savefig(str(ytitle)+'.png')

def nn_test(data, labels, params):
    h, output, cost = forward_prop(data, labels, params)
    accuracy = compute_accuracy(output, labels)
    return accuracy

def compute_accuracy(output, labels):
    accuracy = (np.argmax(output,axis=1) == np.argmax(labels,axis=1)).sum() * 1. / labels.shape[0]
    return accuracy

def one_hot_labels(labels):
    one_hot_labels = np.zeros((labels.size, 10))
    one_hot_labels[np.arange(labels.size),labels.astype(int)] = 1
    return one_hot_labels

def main():
    np.random.seed(100)
    trainData, trainLabels = readData('images_train.csv', 'labels_train.csv')
    trainLabels = one_hot_labels(trainLabels)
    p = np.random.permutation(60000)
    trainData = trainData[p,:]
    trainLabels = trainLabels[p,:]

    devData = trainData[0:10000,:]
    devLabels = trainLabels[0:10000,:]
    trainData = trainData[10000:,:]
    trainLabels = trainLabels[10000:,:]

    mean = np.mean(trainData)
    std = np.std(trainData)
    trainData = (trainData - mean) / std
    devData = (devData - mean) / std

    testData, testLabels = readData('images_test.csv', 'labels_test.csv')
    testLabels = one_hot_labels(testLabels)
    testData = (testData - mean) / std
	
    params = nn_train(trainData, trainLabels, devData, devLabels)


    readyForTesting = False
    if readyForTesting:
        accuracy = nn_test(testData, testLabels, params)
	print 'Test accuracy: %f' % accuracy

if __name__ == '__main__':
    main()
