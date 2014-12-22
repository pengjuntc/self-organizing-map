import numpy as np
from sys import argv  
import random
import matplotlib.pyplot as plt
import math

def myPlot(matrixX, matrixY, N, name, index):
    fig = plt.figure(index)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(name)
    plt.scatter(dataX, dataY, c = 'b')
    plt.scatter(weightsX, weightsY, c = 'r')
    for i in range(0, N):
        plt.plot(matrixX[N*i:N*(i+1)], matrixY[N*i:N*(i+1)], c='g')
        plt.plot(matrixX[i::N], matrixY[i::N], c = 'y')
    plt.show()
    name.replace(" ", "")
    fig.savefig(name, ext = "png")

script, training_file, testing_file = argv

# get training file and testing file 
training = np.loadtxt(training_file)
testing = np.loadtxt(testing_file)

trainingX = training[:, 2].tolist()
trainingY = training[:, 3].tolist()
testingX = testing[:, 2].tolist()
testingY = testing[:, 3].tolist()

# merge training dataset and testing dataset
dataX = trainingX + testingX
dataY = trainingY + testingY
dataLen = len(dataX)

# max iterations
MAX_ITERATIONS = 3000

# N*N output layer network topology 
N = 32

# generate random weights for output layer
weightsX = [random.random() for i in range(N*N)]
weightsY = [random.random() for i in range(N*N)]

# initial sigma value
sigma0 = (N-1) * (2**0.5)

# suggested tau value
tau = MAX_ITERATIONS / math.log(sigma0)
#tau = MAX_ITERATIONS / sigma0

#initial learning rate
eta0 = 0.1

figNumber = 0
wChange = []

# training process
for i in range(MAX_ITERATIONS):
    # pick data to training network at random
    index = random.randint(0, dataLen-1)
    # index for winner neuron
    winnerIndex = 0
    minDist = None
    for j in range(N*N):
        # Euclidean distance for weights vector
        dist = ( (weightsX[j] - dataX[index])**2 + (weightsY[j] - dataY[index])**2 )**0.5
        if minDist is None or dist < minDist:
            minDist = dist
            winnerIndex = j
    
    # corresponding row and column index for winner neuron
    winnerRow = winnerIndex / N
    winnerCol = winnerIndex % N

    # calculate the learning rate
    eta = eta0 * math.exp(-i/float(MAX_ITERATIONS))
    if eta < 0.01:
        eta = 0.01
    
    # calculate the sigma value
    sigma = sigma0 * math.exp(-i/tau)
    
    # weight change per iteration
    weightChange = 0
    for j in range(N*N):
        currRow = j / N
        currCol = j % N
        distToWinner = ((currRow - winnerRow)**2 +(currCol - winnerCol)**2 )**0.5
        neighborFunc = math.exp(-distToWinner**2 / (2 * (sigma)**2))
        xChange =  eta*neighborFunc*(dataX[index] - weightsX[j])
        yChange =  eta*neighborFunc*(dataY[index] - weightsY[j])
        weightsX[j] = weightsX[j] + xChange
        weightsY[j] = weightsY[j] + yChange
        weightChange += 0.5 * (xChange**2 + yChange**2)**0.5
    
    weightChange /= N*N
    wChange.append(weightChange)

    if i%500 ==0 :
        if i == 0:
            name = str(i) + ' iteration'
        else:
            name = str(i) + ' iterations'
        name = str(N) + '*' + str(N) + ' topology ' + name
        myPlot(weightsX, weightsY, N, name, figNumber)
        figNumber += 1

name = str(N) + '*' + str(N) + ' topology ' + str(MAX_ITERATIONS) + ' iterations'
myPlot(weightsX, weightsY, N, name, figNumber)
#figNumber += 1

fig = plt.figure(figsize = (10, 5), dpi = 80)
x = [i for i in range(MAX_ITERATIONS)]
name = str(N) + '*' + str(N) + ' topology weight change VS iterations'
plt.title(name)
plt.xlabel('Iterations')
plt.ylabel('Weight Change')
plt.xlim(0, MAX_ITERATIONS)
plt.scatter(x, wChange, c = 'g', marker = 's')
name.replace(" ", "")
fig.savefig(name, ext = 'png')
plt.show()

