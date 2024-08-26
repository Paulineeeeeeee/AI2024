# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        foodDistances = [manhattanDistance(newPos, foodPos) for foodPos in newFood.asList()]
        # 計算新位置到所有鬼魂的距離
        ghostDistances = [manhattanDistance(newPos, ghostState.getPosition()) for ghostState in newGhostStates]

        # 計算鬼魂恐懼時間的最小值，以判斷是否有安全的進攻機會
        minScaredTime = min(newScaredTimes) if newScaredTimes else 0

        # 如果有鬼魂非常接近（距離小於2），則給予非常低的評分
        if ghostDistances and min(ghostDistances) < 2:
            return -float('inf')

        # 評估函數的分數基於以下幾點：
        # - 更靠近食物的狀態更好
        # - 如果有安全的進攻機會（即鬼魂處於恐懼狀態），該狀態更好
        # 使用食物距離的倒數和鬼魂恐懼時間來計算評估分數，並給予不同的權重
        score = successorGameState.getScore()
        score += sum([1.0 / dist for dist in foodDistances]) if foodDistances else 0
        score += minScaredTime * 2  # 給予鬼魂恐懼時間的權重

        return score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()       
        # 定義最大值函數
        def maxValue(gameState, depth, agentIndex):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            v = float("-inf")
            for action in gameState.getLegalActions(agentIndex):
                v = max(v, minValue(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1))
            return v

        # 定義最小值函數
        def minValue(gameState, depth, agentIndex):
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            v = float("inf")
            nextAgentIndex = agentIndex + 1
            if nextAgentIndex >= gameState.getNumAgents():
                nextAgentIndex = 0
                depth -= 1
            if nextAgentIndex == 0:
                func = maxValue
            else:
                func = minValue
            for action in gameState.getLegalActions(agentIndex):
                v = min(v, func(gameState.generateSuccessor(agentIndex, action), depth, nextAgentIndex))
            return v

        # 遊戲的主要決策進程，從這裡開始
        bestScore = float("-inf")
        bestAction = None
        for action in gameState.getLegalActions(0):
            score = minValue(gameState.generateSuccessor(0, action), self.depth, 1)
            if score > bestScore:
                bestScore = score
                bestAction = action
        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def maxValue(gameState, depth, agentIndex, alpha, beta):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            v = float("-inf")
            for action in gameState.getLegalActions(agentIndex):
                v = max(v, minValue(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1, alpha, beta))
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v

        # 定義最小值函數，加入alpha和beta
        def minValue(gameState, depth, agentIndex, alpha, beta):
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            v = float("inf")
            nextAgentIndex = agentIndex + 1
            if nextAgentIndex >= gameState.getNumAgents():
                nextAgentIndex = 0
                depth -= 1
            if nextAgentIndex == 0:
                func = lambda gameState, depth, agentIndex, alpha, beta: maxValue(gameState, depth, agentIndex, alpha, beta)
            else:
                func = lambda gameState, depth, agentIndex, alpha, beta: minValue(gameState, depth, agentIndex, alpha, beta)
            for action in gameState.getLegalActions(agentIndex):
                v = min(v, func(gameState.generateSuccessor(agentIndex, action), depth, nextAgentIndex, alpha, beta))
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v

        # 主體算法
        bestScore = float("-inf")
        bestAction = None
        alpha = float("-inf")
        beta = float("inf")
        for action in gameState.getLegalActions(0):
            score = minValue(gameState.generateSuccessor(0, action), self.depth, 1, alpha, beta)
            if score > bestScore:
                bestScore = score
                bestAction = action
            alpha = max(alpha, score)
        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
