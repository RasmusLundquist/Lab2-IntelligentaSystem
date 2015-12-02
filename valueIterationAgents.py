import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        #self.startState = mdp.getStartState()
        #self.newCounter = util.Counter()

        # Write value iteration code here
        "*** YOUR CODE HERE ***"


        for i in range(self.iterations):
            newCounter = self.values.copy()
            states = self.mdp.getStates()
            for state in states:
                highestValue = None
                legalActions = self.mdp.getPossibleActions(state)
                for action in legalActions:
                    temp = self.getQValue(state, action)
                    if highestValue < temp:
                        highestValue = temp
                if highestValue is None:
                    highestValue = 0
                newCounter[state] = highestValue
            self.values = newCounter





    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        value = 0.0
        successors = self.mdp.getTransitionStatesAndProbs(state, action)

        for successorState, prob in successors:
            value += prob * (self.mdp.getReward(state,action,successorState) + (self.discount * self.values[successorState]))
        return value


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        bestAction = float("-inf"), None
        legalActions = self.mdp.getPossibleActions(state)
        if len(legalActions) == 0:
            return None

        for action in legalActions:
            temp = self.getQValue(state, action), action
            if bestAction[0] < temp[0]:
                bestAction = temp
        return bestAction[1]

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
