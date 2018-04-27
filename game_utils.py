#!/usr/bin/python

import numpy as np
import copy
import random
import os
import numpy as np
import time
import sys

from collections      import deque
from keras.models     import Sequential
from keras.layers     import Dense, Conv2D, BatchNormalization
from keras.optimizers import Adam, SGD
from IPython.display import clear_output
from datetime import timedelta
from keras.layers.core import Dropout
from sklearn.preprocessing import normalize
from keras.layers import Activation
from keras.layers.advanced_activations import LeakyReLU


CIRCLE = 0.5
CROSS = 1.0
DEBUG = False

def rewardFromCloseness(board, action, piece):
    row = int(action/board.shape[1]) + 1
    col = action % board.shape[0] + 1
    if(DEBUG == True):
        print("Action: ", action, "Row/Col ", row, col)
        print(board.shape)
    board = np.pad(board, (1,1), mode='constant')
    distance = 1
    neighbours = board[row-distance:row+distance+1, col-distance:col+distance+1].flatten()
    neighbours[4] = 0.0
    if(DEBUG == True):
        nprint = np.reshape(neighbours, (3,3))
        print(nprint)
    reward = 0.0
    if(np.count_nonzero(neighbours == piece) > 0):
        reward += 2.0
    if(np.count_nonzero(neighbours == nextPiece(piece)) > 0):
        reward += 1.0
    if(DEBUG == True):
        print(reward)
    return reward
    
def switchPiecesOfState(state):
    switchedState = []
    for x in state[0]:
        value = 0.0
        if(x == CIRCLE):
            value = CROSS
        elif(x == CROSS):
            value = CIRCLE
        switchedState.append(value)

    return np.reshape(np.array(switchedState), [1, len(switchedState)])  
                                       
def nextPiece(currentPiece):
    if (currentPiece == CIRCLE):
        return CROSS
    elif (currentPiece == CROSS):
        return CIRCLE

def printBoard(board):
    flat_board = copy.deepcopy(board)
    flat_board = flat_board.flatten()
    printBoard = []
    for x in flat_board:
        value = '.'
        if(x == CIRCLE):
            value = 'O'
        elif(x == CROSS):
            value = 'X'
        printBoard.append(value)
    print(np.array(printBoard).reshape(15,15))
    
def checkWinningCondition(board):
    #Check all rows for 5 consecutive pieces of the same type
    
    for i in range(board.shape[0]):
        for j in range(board.shape[1]-4):
            horizontal = board[i][j:j+5]
            if(horizontal[0] != 0 and all(x==horizontal[0] for x in horizontal)):
                return True

    #Check all columns for 5 consecutive pieces of the same type
    
    for i in range(board.shape[0]-4):
        for j in range(board.shape[1]):
            vertical = [board[i+k][j] for k in range(5)]
            if(vertical[0] != 0 and all(x==vertical[0] for x in vertical)):
                return True
            
    #Check for 5 consecutive pieces of the same time diagonally
    
    for i in range(board.shape[0]-4):
        for j in range(board.shape[1]-4):
            diagonal = [board[i+k][j+k] for k in range(5)]
            if(diagonal[0] != 0 and all(x==diagonal[0] for x in diagonal)):
                return True
            
    for i in range(4,board.shape[0]):
        for j in range(board.shape[1]-4):
            diagonal = [board[i-k][j+k] for k in range(5)]
            if(diagonal[0] != 0 and all(x==diagonal[0] for x in diagonal)):
                return True
            
    return False

def isBoardFull(board):
    if 0 in board[:,:]:
        return False
    return True

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1) 

class Agent():
    def __init__(self, stateSize, actionSize, loadModel, parameterList):
        self.weights            = "fir_weight.h5"
        self.stateSize          = stateSize
        self.actionSize        = actionSize
        self.memory             = deque(maxlen=2000)
        self.learningRate      = 0.001
        self.gamma              = 0.95
        self.explorationRate   = 1.0
        self.explorationMin    = 0.01
        self.explorationDecay  = 0.995
        self.loadModel          = loadModel
        self.brain              = self._buildModel(parameterList)
        self.loss               = deque(maxlen=100)
    
    def _buildModel(self, parameterList):
        model = Sequential()
        model.add(Dense(1024, input_dim=self.stateSize))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dropout(0.3))
        model.add(Dense(1024))     
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dropout(0.3))

        opt = SGD(lr=self.learningRate , momentum=0.9, decay=1e-18, nesterov=False)
        adam = Adam(lr = 0.001)
       
        model.add(Dense(self.actionSize, activation='linear'))
        model.compile(loss='mse', optimizer=opt)
        
        if(self.loadModel == True):
            if os.path.isfile(self.weights):
                model.load_weights(self.weights)
                self.explorationRate = self.explorationMin
        if(parameterList is not None):
            model = parameterList['model']
            model.compile(loss='mse', optimizer = parameterList['opt'])
        return model

    def saveModel(self):
            self.brain.save(self.weights)

    def act(self, state):
        if np.random.rand() <= self.explorationRate:
            availableMoves = np.where(state[0] == 0.0)[0]
            actionValues = np.random.choice(availableMoves)
            return actionValues
        
        actionValues = self.brain.predict(state)
        actionValues = self.maskInvalidActions(actionValues, state)

        return np.argmax(actionValues[0])
    
    def actWithoutExploration(self, state):
        actionValues = self.brain.predict(state)
        actionValues = self.maskInvalidActions(actionValues, state)
        return np.argmax(actionValues[0])
    
    def remember(self, state, action, reward, nextState, done):
        self.memory.append((state, action, reward, nextState, done))

    def rememberLoss(self,loss):
        self.loss.append(loss)
        
    def replay(self, sampleBatchSize):
        if len(self.memory) < sampleBatchSize:
            return
        sampleBatch = random.sample(self.memory, sampleBatchSize)
        X_train = []
        y_train = []
        for state, action, reward, nextState, done in sampleBatch:
            update = reward
            if not done:
                newMaxQ = np.amax(self.brain.predict(nextState)[0])
                update = reward + self.gamma * newMaxQ
            y = self.brain.predict(state)
            y[0][action] = update
            if(np.isnan(y[0]).any()):
                sys.exit()
            X_train.append(state.reshape((self.stateSize,)))
            y_train.append(y.reshape((self.actionSize,)))
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        self.brain.fit(X_train, y_train, epochs = 1, verbose = 0, batch_size = sampleBatchSize)
        if self.explorationRate > self.explorationMin:
            self.explorationRate *= self.explorationDecay
    
    def maskInvalidActions(self, actions, state):
        invalidActionMask = np.full_like(actions, 0.0)
        for i in range (self.actionSize):
            if(state[0][i] != 0.0):
                invalidActionMask[0][i] = float("-inf")

        actionValues = actions+invalidActionMask
        return actionValues


class Environment():
    def __init__(self, size):
        self.board = np.zeros((size,size))
        self.boardSize = size
        self.stateSize = size**2
        self.actionSize = size**2
        self.nextPiece = CROSS

        
    def makeMove(self,action):
        if(isBoardFull(self.board) or checkWinningCondition(self.board)):
            printBoard(self.board)
        state =  np.reshape(self.board, [1, self.stateSize])       

        if(self.nextPiece == CROSS):
            state[0][action] = CROSS
        else:
            state[0][action] = CIRCLE
        
        closenessReward = rewardFromCloseness(self.board, action, self.nextPiece)
        
        self.nextPiece = nextPiece(self.nextPiece)
        self.board = np.reshape(state, (self.boardSize, self.boardSize))
        nextState = self.board
        reward = 0.0
        done = False

        if(checkWinningCondition(self.board)):
            reward += 400.0
            done = True
        if(isBoardFull(self.board)):
            reward += -5.0
            done = True
        else:
            reward -= 0.01
        
        reward += closenessReward
        return nextState, reward, done
    
    def reset(self):
        self.board = np.zeros((self.boardSize, self.boardSize))
        self.nextPiece = CROSS

        
    def opponentMove(self,opponentAction):
        nextState,reward,done = self.makeMove(opponentAction)
        state = np.reshape(nextState, [1, self.stateSize])
        return state, done
            
class FIR():
    def __init__(self, learningPiece, loadModel, parameterList, epochs = 2500):
        self.sampleBatchSize = 32
        self.epochs = epochs
        self.environment = Environment(15)
        self.stateSize = self.environment.stateSize
        self.actionSize = self.environment.actionSize
        self.agent = Agent(self.stateSize, self.actionSize, loadModel, parameterList)
        self.learningPiece = learningPiece
    def test(self):
        DEBUG = True
        done = False
        count = 0
        self.environment = Environment(15)
        state = copy.deepcopy(self.environment.board)
        state = np.reshape(state, [1,self.stateSize])

        while not done:
            action = self.agent.actWithoutExploration(state)
            nextState, reward, done = self.environment.makeMove(action)
            
            printBoard(nextState)
            print(reward)
            print('----------------------------------------------------------------------------')
            state = np.reshape(nextState, [1, self.stateSize])
            count += 1
            if(count > self.actionSize):
                break
        print("Total steps: ", count)
            
    def run(self):
        DEBUG = False
        try:
            start_time = time.time()
            for epoch in range(self.epochs):
                self.environment.reset()
                state = copy.deepcopy(self.environment.board)
                state = np.reshape(state, [1,self.stateSize])
                done = False
                stepIndex = 0
                while not done:
                    stepIndex += 1
                    if(stepIndex > self.actionSize + 1):
                        print("Something is wrong..")
                        done = True
                        break
                    if(self.environment.nextPiece != self.learningPiece):
                        opponentAction = 0
                        if(self.environment.nextPiece == CIRCLE):
                            switchedState = switchPiecesOfState(state)
                            opponentAction = self.agent.actWithoutExploration(switchedState)
                        else:
                            opponentAction = self.agent.actWithoutExploration(state)

                        state, done = self.environment.opponentMove(opponentAction)
                        if(done == True):
                            break
                    else:
                        switchBackNeeded = False
                        action = 0
                        if(self.environment.nextPiece == CIRCLE):
                            switchedState = switchPiecesOfState(state)
                            action = self.agent.act(switchedState)
                            switchBackNeeded = True
                        else:
                            action = self.agent.act(state)


                        nextState, reward, done = self.environment.makeMove(action)
                        nextState = np.reshape(nextState, [1, self.stateSize])
                        if(switchBackNeeded == True):
                            self.agent.remember(switchPiecesOfState(state), action, reward, switchPiecesOfState(nextState), done)
                        else:
                            self.agent.remember(state, action, reward, nextState, done)
                            
                        state = nextState
                        
                current_time = time.time()
                time_dif = current_time - start_time
                if((epoch + 1) % 100 == 0):
                    print("Epoch: ", epoch + 1,"/", self.epochs, "\tElapsed time: ",  str(timedelta(seconds=int(round(time_dif)))))
                          #, "\t Loss:", sum(self.agent.loss)/len(self.agent.loss))
                self.agent.replay(self.sampleBatchSize)
                self.learningPiece = nextPiece(self.learningPiece)
        finally:
            
            self.agent.saveModel()
        