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
        if successorGameState.isLose():
            return float("-inf")

        score = successorGameState.getScore()

        if action == Directions.STOP:
            score -= 5

        foodList = newFood.asList()
        if foodList:
            minFoodDist = min(manhattanDistance(newPos, foodPos) for foodPos in foodList)
            score += 10.0 / (minFoodDist + 1.0)

        for ghostState, scaredTime in zip(newGhostStates, newScaredTimes):
            ghostPos = ghostState.getPosition()
            dist = manhattanDistance(newPos, ghostPos)

            if scaredTime > 0:
                if dist > 0:
                    score += 5.0 / dist
            else:
                if dist == 0:
                    return float("-inf")
                if dist <= 1:
                    score -= 200
                else:
                    score -= 2.0 / dist

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
        def minimax(state, agentIndex, depth):
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            numAgents = state.getNumAgents()

            if agentIndex == 0:
                bestValue = float("-inf")
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    value = minimax(successor, 1, depth)
                    if value > bestValue:
                        bestValue = value
                return bestValue

            else:
                bestValue = float("inf")
                nextAgent = agentIndex + 1
                nextDepth = depth
                if nextAgent == numAgents:
                    nextAgent = 0
                    nextDepth = depth + 1

                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    value = minimax(successor, nextAgent, nextDepth)
                    if value < bestValue:
                        bestValue = value
                return bestValue

        bestAction = None
        bestValue = float("-inf")

        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            value = minimax(successor, 1, 0)

            if value > bestValue:
                bestValue = value
                bestAction = action

        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alphabeta(state, agentIndex, depth, alpha, beta):
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            numAgents = state.getNumAgents()

            if agentIndex == 0:
                value = float("-inf")

                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    childValue = alphabeta(successor, 1, depth, alpha, beta)
                    value = max(value, childValue)

                    alpha = max(alpha, value)

                    if value > beta:
                        return value

                return value

            else:
                value = float("inf")

                nextAgent = agentIndex + 1
                nextDepth = depth
                if nextAgent == numAgents:
                    nextAgent = 0
                    nextDepth = depth + 1

                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    childValue = alphabeta(successor, nextAgent, nextDepth, alpha, beta)
                    value = min(value, childValue)

                    beta = min(beta, value)

                    if value < alpha:
                        return value

                return value

        bestAction = None
        bestValue = float("-inf")
        alpha = float("-inf")
        beta = float("inf")

        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            value = alphabeta(successor, 1, 0, alpha, beta)

            if value > bestValue:
                bestValue = value
                bestAction = action

            alpha = max(alpha, bestValue)

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
        def expectimax(state, agentIndex, depth):
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            numAgents = state.getNumAgents()
            legalActions = state.getLegalActions(agentIndex)

            if not legalActions:
                return self.evaluationFunction(state)

            if agentIndex == 0:
                bestValue = float("-inf")
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value = expectimax(successor, 1, depth)
                    if value > bestValue:
                        bestValue = value
                return bestValue

            else:
                nextAgent = agentIndex + 1
                nextDepth = depth
                if nextAgent == numAgents:
                    nextAgent = 0
                    nextDepth = depth + 1

                prob = 1.0 / len(legalActions)
                expectedValue = 0.0

                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value = expectimax(successor, nextAgent, nextDepth)
                    expectedValue += prob * value

                return expectedValue

        bestAction = None
        bestValue = float("-inf")

        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            value = expectimax(successor, 1, 0)
            if value > bestValue:
                bestValue = value
                bestAction = action

        return bestAction

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:
    Combines Pacman's score with weighted features:
    - rewards being closer to food and capsules,
    - penalizes distance to active (non-scared) ghosts,
    - rewards approaching scared ghosts,
    - penalizes remaining food and capsules.
    Designed to remain smooth (no infinities) to prevent expectimax from crashing.
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isWin():
        return float("inf")
    if currentGameState.isLose():
        return float("-inf")

    pos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()

    score = float(currentGameState.getScore())
    value = score

    foodCount = len(foodList)
    value -= 3.0 * foodCount

    if foodList:
        minFoodDist = min(manhattanDistance(pos, f) for f in foodList)
        value += 3.0 / (minFoodDist + 1.0)

    dangerPenalty = 0
    scaredBonus = 0

    for ghost in ghostStates:
        gpos = ghost.getPosition()
        dist = manhattanDistance(pos, gpos)

        if ghost.scaredTimer > 0:
            if dist > 0:
                scaredBonus += 5.0 / dist
        else:
            if dist == 0:
                dangerPenalty += 500
            else:
                dangerPenalty += 10.0 / dist

    value -= dangerPenalty
    value += scaredBonus

    capCount = len(capsules)
    value -= 4.0 * capCount

    if capsules:
        minCapDist = min(manhattanDistance(pos, c) for c in capsules)
        value += 2.0 / (minCapDist + 1.0)

    return value

# Abbreviation
better = betterEvaluationFunction
