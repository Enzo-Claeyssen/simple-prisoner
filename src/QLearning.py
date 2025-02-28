from strategies import Strategy
from math import exp
import random
import csv


class QLearning(Strategy) :
    """
    QLearning Strategy using QTable
    """
    
    MEMORY_SIZE = 10
    __QTable = [[0 for _ in range(2)] for _ in range(5**MEMORY_SIZE)]
    
    
    def __init__(self, trainingMode, memory = [() for _ in range(MEMORY_SIZE)]) :
        """
        Initialize hyperparameters
        :param: symbole The symbole used by this opponent
        :param: trainingMode True if in trainingMode False otherwise.
        """
        self.name = 'QLearning'
        self.trainingMode = trainingMode
        self.memory = memory
        self.__epsilon = 1
        self.__reducingFactor = 0.99
        self.__decayRate = 0.0005
        self.__numberOfDecay = 0
        self.__learningRate = 0.5
    
    
    def clone(self) :
        return QLearning(self.trainingMode, self.memory)
    
    
    def getAction(self, tick) :
        """
        Returns the action made by the opponent
        :param: state The state where the action is taken.
        :return: An int representing the action
        """
        processed = self.__processState(self.memory)
        if self.trainingMode :
            action = self.__epsilonGreedyPolicy(processed)
        else :
            action = self.__greedyPolicy(processed)
        return self.__translateAction(action)
    
    
    def update(self, my: str, his: str) -> None:
        """
        Called each time to retrieve game's info
        """
        del self.memory[0]
        self.memory.append((my, his))
    
    def finalUpdate(self, myScore, hisScore, myActions, hisActions) :
        if self.trainingMode :
            n = len(myActions)
            
            data = zip(myActions, hisActions)
            for i in range(0, n) :
                state = [() for _ in range(QLearning.MEMORY_SIZE - i)]
                state.extend([(myActions[j], hisActions[j]) for j in range(max(0, i-QLearning.MEMORY_SIZE), i)])
                state = self.__processState(state)
                action = self.__strActionToIntAction(myActions[i])
                reward = myScore * (self.__reducingFactor ** (n-1-i))
                self.learn(state, action, reward)
            
    
    def learn(self, state, action, reward) :
        """
        Updates the QTable
        :param: state The initial state as a grid of Cell where the action has been taken
        :param: action The action that has been made
        :param: reward The immediate reward obtained
        :param: newState The state as a grid of Cell of the env once the other opponent played
        """
        initialState = state
        
        expectedCumulativeReward = QLearning.__QTable[initialState][action]
        error = reward - expectedCumulativeReward
        
        QLearning.__QTable[initialState][action] = expectedCumulativeReward + self.__learningRate * (error)
        self.decayEpsilon()
        
    
    @staticmethod
    def importQTable() :
        """
        Imports the QTable
        """
        newQTable = []
        with open('models/QTableTTT.csv', 'r') as file :
            reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
            newQTable = list(reader)
        QLearning.__QTable = newQTable
    
    @staticmethod
    def exportQTable() :
        """
        Exports the QTable
        """
        with open('models/QTableTTT.csv', 'w') as file :
            writer = csv.writer(file)
            writer.writerows(QLearning.__QTable)
    
    @staticmethod
    def resetQTable() :
        """
        Resets the QTable
        """
        QLearning.__QTable = [[0 for _ in range(2)] for _ in range(5**QLearning.MEMORY_SIZE)]
    
    
    def decayEpsilon(self) :
        """
        Decays the epsilon, less exploration and more exploitation
        """
        self.__epsilon = 0.05 + 0.95 * exp(-1 * self.__decayRate * self.__numberOfDecay)
        self.__numberOfDecay += 1
    
    
    def __greedyPolicy(self, state) :
        """
        Takes an action according to the greedyPolicy.
        :param: state The state as an int where the action has to be taken
        :return: An int describing the action maximizing expected cumulative reward
        """
        possibilities = QLearning.__QTable[state]
        maxi = possibilities[0]
        maxi_i = 0
        for i in range(1, 2) :
            tmp = possibilities[i]
            if tmp > maxi :
                maxi = tmp
                maxi_i = i
        return maxi_i
    
    
    def __epsilonGreedyPolicy(self, state) :
        """
        Takes an action according to the epsilon greedy policy.
        With a probability of epsilon, will take a random action.
        With a probability of 1-epsilon, will act the same way as the greedy policy
        :param: state The state as an int where the action has to be taken.
        :return: An int describing the action
        """
        if random.uniform(0, 1) <= self.__epsilon :
            return random.randint(0, 1)
        else :
            return self.__greedyPolicy(state)
    
    
    def __processSingleState(self, state) :
        if state == () :
            return 0
        elif state == ("D", "D") :
            return 1
        elif state == ("D", "C") :
            return 2
        elif state == ("C", "D") :
            return 3
        elif state == ("C", "C") :
            return 4
        else :
            raise InvalidStateError()
    
    def __processState(self, state) :
        res = 0
        for i in range(len(state)) :
            singleState = state[i]
            res += self.__processSingleState(singleState) * (5**i)
        return res
    
    
    def __translateAction(self, action) :
        if action == 0 :
            return "D"
        else :
            return "C"
        
    def __strActionToIntAction(self, strAction) :
        if strAction == "D" :
            return 0
        else :
            return 1