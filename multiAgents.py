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

from __future__ import division
from util import manhattanDistance
from game import Directions
from game import Actions

import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
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

    def evaluationFunction(self, currentGameState, action):
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
        score = successorGameState.getScore()
        for g in newGhostStates:

            if manhattanDistance(newPos, g.getPosition()) < 2:
                return -999999

        min_dist = 9999999
        for f in newFood.asList():
            dist = manhattanDistance(newPos, f)
            if(dist < min_dist):
                min_dist = dist

        score += 5/min_dist

        return score

def scoreEvaluationFunction(currentGameState):
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

    def getAction(self, gameState):
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
        """
        "*** YOUR CODE HERE ***"

        pacman = 0
        def minimax(gameState, depth, Player):

            if gameState.isWin() or gameState.isLose():
                return gameState.getScore()

            if Player == pacman:

                actions = gameState.getLegalActions(pacman)
                bestValue = float("-inf")

                for action in actions:

                    # depth is checked at ghosts as 1 depth cost means pacman and all ghosts have playes their moves
                    value = minimax(gameState.generateSuccessor(pacman, action), depth, 1)

                    # find the max value and the best move
                    if value > bestValue:
                        bestValue = value
                        bestAction = action

                if depth == 0:
                    return bestAction
                else:
                    return bestValue

            else:

                # if the player is the last ghost the next one(if not max depth) will be pacman
                if Player == gameState.getNumAgents()-1:
                    nxt = pacman
                    dp = depth + 1
                else:
                    nxt = Player + 1
                    dp = depth

                actions = gameState.getLegalActions(Player)
                bestValue = float("inf")

                for action in actions:

                    # if we have reached max depth and all the ghosts have played (thus dp and not depth) play the move and return evaluation
                    if dp == self.depth :
                        value = self.evaluationFunction(gameState.generateSuccessor(Player, action))
                    else:
                        value = minimax(gameState.generateSuccessor(Player, action), dp, nxt)

                    # find the min value
                    bestValue = min(value, bestValue)

                return bestValue

        return minimax(gameState, 0, pacman)
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        pacman = 0
        def alpha_beta(gameState, depth, Player, a, b):

            if gameState.isWin() or gameState.isLose():
                return gameState.getScore()

            if Player == pacman:

                actions = gameState.getLegalActions(pacman)
                bestValue = float("-inf")
                bestAction = Directions.STOP

                for action in actions:

                    # depth is checked at ghosts as 1 depth cost means pacman and all ghosts have playes their moves
                    value = alpha_beta(gameState.generateSuccessor(pacman, action), depth, 1, a, b)

                    # find the max value and the best move
                    if value > bestValue:
                        bestValue = value
                        bestAction = action

                    if bestValue > b:
                        return bestValue

                    a = max(a, bestValue)

                if depth == 0:
                    return bestAction
                else:
                    return bestValue

            else:

                # if the player is the last ghost the next one(if not max depth) will be pacman
                if Player == gameState.getNumAgents()-1:
                    nxt = pacman
                    dp = depth + 1
                else:
                    nxt = Player + 1
                    dp = depth

                actions = gameState.getLegalActions(Player)
                bestValue = float("inf")
                bestAction = Directions.STOP

                for action in actions:

                    # if we have reached max depth and all the ghosts have played (thus dp and not depth) play the move and return evaluation
                    if dp == self.depth :
                        value = self.evaluationFunction(gameState.generateSuccessor(Player, action))
                    else:
                        value = alpha_beta(gameState.generateSuccessor(Player, action), dp, nxt, a, b)

                    # find the min value
                    bestValue = min(value, bestValue)

                    if bestValue < a:
                        return bestValue

                    b = min(b, bestValue)

                return bestValue

        return alpha_beta(gameState, 0, pacman, float("-inf"), float("inf"))
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"

        pacman = 0
        def expectimax(gameState, depth, Player):

            if gameState.isWin() or gameState.isLose():
                return gameState.getScore()

            if Player == pacman:

                actions = gameState.getLegalActions(pacman)
                bestValue = float("-inf")
                bestAction = Directions.STOP
                for action in actions:

                    # depth is checked at ghosts as 1 depth cost means pacman and all ghosts have playes their moves
                    value = expectimax(gameState.generateSuccessor(pacman, action), depth, 1)

                    # find the max value and the best move
                    if value > bestValue:
                        bestValue = value
                        bestAction = action

                if depth == 0:
                    return bestAction
                else:
                    return bestValue

            else:

                # if the player is the last ghost the next one(if not max depth) will be pacman
                if Player == gameState.getNumAgents()-1:
                    nxt = pacman
                    dp = depth + 1
                else:
                    nxt = Player + 1
                    dp = depth

                actions = gameState.getLegalActions(Player)
                value = 0

                for action in actions:
                    # if we have reached max depth and all the ghosts have played (thus dp and not depth) play the move and return evaluation
                    if dp == self.depth :
                        value = self.evaluationFunction(gameState.generateSuccessor(Player, action))
                    else:
                        value += 1/len(actions) * expectimax(gameState.generateSuccessor(Player, action), dp, nxt)

                return value

        return expectimax(gameState, 0, pacman)
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    capsules = currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    foodNum = len(food.asList())
    clf = fsum = csum = gsum = 0

    score = currentGameState.getScore()

    # consider ghost positions and number
    for g in ghostStates:
        g_dist = []
        dist = manhattanDistance(pos, g.getPosition())

        if g.scaredTimer > 8:
            if dist == 0 :
                g_dist.append(float("+inf"))
            elif dist < 2 :
                g_dist.append(80/dist)
            elif dist < 3 :
                g_dist.append(60/dist)
            elif dist < 4 :
                g_dist.append(40/dist)
        else:
            if dist > 1 :
                g_dist.append(-(1/ dist))
            else :
                g_dist.append(float("-inf"))
        gsum = sum(g_dist)

    # consider capsule position and number and find the closest capsule
    for c in capsules:
        c_dist = []
        dist = manhattanDistance(pos, c)
        if dist > 5 :
            c_dist.append(1/ dist)
        elif dist > 4 :
            c_dist.append(10/ dist)
        elif dist > 3 :
            c_dist.append(30/ dist)
        elif dist > 2 :
            c_dist.append(40/ dist)
        elif dist > 0 :
            c_dist.append(999/dist)
        csum = sum(c_dist)

    # consider food position and number and find the closest food
    for f in food.asList():
        f_dist = []
        dist = manhattanDistance(pos, f)
        if dist > 0 :
            f_dist.append(1/ dist)
        # else : f_dist.append(10)
        fsum = sum(f_dist)
        clf = min(f_dist)

    score += gsum
    score += fsum
    # score += csum

    return score

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
