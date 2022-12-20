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
        newGhostPositions = successorGameState.getGhostPositions()
        newFoodList = newFood.asList()
        return_score = successorGameState.getScore()
        
        def find_dist_of_closest(init_pos,list_of_pos):
            dist_list = []
            for pos in list_of_pos:
                manh_dist = util.manhattanDistance(pos, init_pos)
                dist_list.append(manh_dist)
            if(len(dist_list)!=0):
                min_of_dist_list = min(dist_list)
                if(min_of_dist_list!=0):
                    return min_of_dist_list
                else:
                    return -1
            else:
                return -1
        def both_ghost_scared(time):
            for t in time:
                if(t!=0):
                    return False
            return True
        
        evfunc_food_dist = find_dist_of_closest(newPos, newFoodList)
        evfunc_ghost_dist = find_dist_of_closest(newPos, newGhostPositions)
        
        if(evfunc_food_dist!=-1):
            return_score += 1.0/(evfunc_food_dist)
        if(evfunc_ghost_dist!=-1 and not both_ghost_scared(newScaredTimes)):
            return_score -= 1.0/(evfunc_ghost_dist)
        if(evfunc_ghost_dist<2 and not both_ghost_scared(newScaredTimes)):
            return -float("inf")
        
        return return_score     
    
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def max_value(state, agentIndex, depth):
            agentIndex = 0
            legal_actions = state.getLegalActions(agentIndex)
            
            if not legal_actions  or depth == self.depth:
                return self.evaluationFunction(state)
            
            v = -float("inf")
            for action in legal_actions:
                next_successors = state.generateSuccessor(agentIndex, action)
                v =  max(v,min_value(next_successors, agentIndex+1,  depth+1))
            return v
        
        def min_value(state, agentIndex, depth):
            agent_num = gameState.getNumAgents()
            legal_actions = state.getLegalActions(agentIndex)

            if not legal_actions:
                return self.evaluationFunction(state)

            v = float("inf")
            for action in legal_actions:
                next_successors = state.generateSuccessor(agentIndex, action)
                if agentIndex == agent_num - 1:
                    v =  min(v,max_value(next_successors, agentIndex,  depth))
                else:
                    v =  min(v,min_value(next_successors, agentIndex+1,  depth))
            return v
        
        legal_actions = gameState.getLegalActions(0)
    
        first_depth_minimax = []
        
        for action in legal_actions:
            first_depth_successors = gameState.generateSuccessor(0,action)
            first_depth_minimax.append(min_value(first_depth_successors,1,1))
        
        max_of_minimax = max(first_depth_minimax)
        best_indices = [index for index in range(len(first_depth_minimax)) if first_depth_minimax[index] == max_of_minimax]
        chosen_index = random.choice(best_indices)
        
        return legal_actions[chosen_index]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def max_value(state, agentIndex, depth,a,b):
            agentIndex = 0
            legal_actions = state.getLegalActions(agentIndex)

            if not legal_actions  or depth == self.depth:
                return self.evaluationFunction(state)
            
            v = -float("inf")
            for action in legal_actions:
                next_successors = state.generateSuccessor(agentIndex, action)
                v =  max(v,min_value(next_successors, agentIndex+1,  depth+1,a,b))
                if(v>b):
                    return v
                a = max(a,v)
            return v
        
        def min_value(state, agentIndex, depth,a,b):
            agent_num = gameState.getNumAgents()
            legal_actions = state.getLegalActions(agentIndex)

            if not legal_actions:
                return self.evaluationFunction(state)

            v = float("inf")
            for action in legal_actions:
                next_successors = state.generateSuccessor(agentIndex, action)
                if agentIndex == agent_num - 1:
                    v =  min(v,max_value(next_successors, agentIndex, depth, a, b))
                    if(v<a):
                        return v
                else:
                    v =  min(v,min_value(next_successors, agentIndex+1, depth, a, b))
                    if(v<a):
                        return v
                b = min(b,v)
            return v
        
        legal_actions = gameState.getLegalActions(0)
    
        first_depth_minimax = []
        a = -float('inf')
        b = float('inf')
        v = -float('inf')
        for action in legal_actions:
            first_depth_successors = gameState.generateSuccessor(0,action)
            minimax=min_value(first_depth_successors,1,1,a,b)
            first_depth_minimax.append(minimax)
            if minimax > v:
                v = minimax
            a = max(a, minimax)
            
        best_indices = [index for index in range(len(first_depth_minimax)) if first_depth_minimax[index] == v]
        chosen_index = random.choice(best_indices)
        
        return legal_actions[chosen_index]

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
        def max_value(state, agentIndex, depth):
            agentIndex = 0
            legal_actions = state.getLegalActions(agentIndex)

            if not legal_actions  or depth == self.depth:
                return self.evaluationFunction(state)
            
            v = -float("inf")
            for action in legal_actions:
                next_successors = state.generateSuccessor(agentIndex, action)
                v =  max(v,exp_value(next_successors, agentIndex+1,  depth+1))
            return v
        
        def exp_value(state, agentIndex, depth):
            agent_num = gameState.getNumAgents()
            legal_actions = state.getLegalActions(agentIndex)

            if not legal_actions:
                return self.evaluationFunction(state)
            
            v = 0
            for action in legal_actions:
                next_successors = state.generateSuccessor(agentIndex, action)
                probability = 1.0/len(legal_actions)
                if agentIndex == agent_num - 1:
                    v += probability*max_value(next_successors, agentIndex,  depth)
                else:
                    v += probability*exp_value(next_successors, agentIndex+1,  depth)
            return v
        
        legal_actions = gameState.getLegalActions(0)
    
        first_depth_expectimax = []
        
        for action in legal_actions:
            first_depth_successors = gameState.generateSuccessor(0,action)
            first_depth_expectimax.append(exp_value(first_depth_successors,1,1))
        
        max_of_expectimax = max(first_depth_expectimax)
        best_indices = [index for index in range(len(first_depth_expectimax)) if first_depth_expectimax[index] == max_of_expectimax]
        chosen_index = random.choice(best_indices)
        
        return legal_actions[chosen_index]

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    
    "*** YOUR CODE HERE ***"
    """
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    newGhostPositions = currentGameState.getGhostPositions()
    newFoodList = newFood.asList()
    capsules = currentGameState.getCapsules()                             ####
    return_score = currentGameState.getScore()
    
    def find_dist_of_closest(init_pos,list_of_pos):
        dist_list = []
        for pos in list_of_pos:
            manh_dist = util.manhattanDistance(pos, init_pos)
            dist_list.append(manh_dist)
        if(len(dist_list)!=0):
            min_of_dist_list = min(dist_list)
            if(min_of_dist_list!=0):
                return min_of_dist_list
            else:
                return -1
        else:
            return -1
    def both_ghost_scared(time):
        for t in time:
            if(t!=0):
                return False
        return True
    
    evfunc_food_dist = find_dist_of_closest(newPos, newFoodList)
    evfunc_ghost_dist = find_dist_of_closest(newPos, newGhostPositions)
    evfunc_capsule_dist = find_dist_of_closest(newPos, capsules)          ####
        
    if(evfunc_capsule_dist!=-1):                                          ####
        return_score += 2.0/(evfunc_capsule_dist)
    if(evfunc_food_dist!=-1):
        return_score += 1.0/(evfunc_food_dist)
    if(evfunc_ghost_dist!=-1 and not both_ghost_scared(newScaredTimes)):
        return_score -= 1.0/(evfunc_ghost_dist)
    if(evfunc_ghost_dist<2 and not both_ghost_scared(newScaredTimes)):
        return -float("inf")
    
    return return_score

# Abbreviation
better = betterEvaluationFunction
