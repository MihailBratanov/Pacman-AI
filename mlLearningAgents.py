# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
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

# The agent here was written by Simon Parsons, based on the code in
# pacmanAgents.py
# learningAgents.py

from pacman import Directions
from game import Agent
import random
import game
import util


# QLearnAgent
# The agent vaguely resembles the banditAgent from week 8 in terms of logic.
# This one however implements Q-Learning using the Bellman equation and alpha 
# decay as per the lecture. Some code was reused from the banditAgent which is
# stated where necessary. 
class QLearnAgent(Agent):

    # Constructor, called when we start running the
    def __init__(self, alpha=0.2, epsilon=0.05, gamma=0.8, numTraining = 10):
        # alpha       - learning rate
        # epsilon     - exploration rate
        # gamma       - discount factor
        # numTraining - number of training episodes
        #
        # These values are either passed from the command line or are
        # set to the default values above. We need to create and set
        # variables for them
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.numTraining = int(numTraining)
        # Count the number of games we have played 
        self.episodesSoFar = 0

        #initalize the Q table
        self.q_table= {}
        #Keep track of the previous getAction() parameters 
        self.last_score=0
        self.old_state=[]
        #alphas for the alpha decay, we will have one list of alphas for the entire training period 
        self.alphas=[]
    
    # Accessor functions for the variable episodesSoFars controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar +=1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value):
        self.epsilon = value

    def getAlpha(self):
        return self.alpha

    def setAlpha(self, value):
        self.alpha = value
        
    def getGamma(self):
        return self.gamma

    #inspired by banditAgents.py, in the coursework for week 8,
    #here we are saving the current game score to access in the next getAction()
    def getLastScore(self):
        return self.last_score

    def setLastScore(self, score):
        self.last_score=score

    #Q table accessor method
    def getQVal(self,state, action):
        #returns the q value stored for this state, action pair or 0 otherwise
        if (state, action) in self.q_table:

            return self.q_table[(state,action)] 
        else: 
            return 0.0       

    def getMaxRewardAction(self, state,legal):
        #returns the legal action with the best q-value or None if there is no more legal moves
        #if there is no best value we return a random legal action to prevent crashing 
        values=[]
        if len(legal)==0:
            return None
        for action in legal:
            values.append(self.getQVal(state,action))
        if len(values)==0:
            return random.choice(legal)
        else:
            best_value=max(values)
            best_value_idx=values.index(best_value)
            return legal[best_value_idx]

    def pick_action(self, state):
        #greedily decides how pacman will act
        legal_actions=state.getLegalPacmanActions()
        if Directions.STOP in legal_actions:
            legal_actions.remove(Directions.STOP)
        if random.random()<=(1-self.epsilon):
            #exploit
            return self.getMaxRewardAction(state,legal_actions)
        else:
            #explore
            return random.choice(legal_actions)

    def getMaxQValue(self, state):
        #returns the current best Q-value for a state from the table; 
        #used in the Bellman equation 
        values=[]
        legal_actions=state.getLegalPacmanActions()
        if Directions.STOP in legal_actions:
            legal_actions.remove(Directions.STOP)
        for action in legal_actions:
            values.append(self.getQVal(state,action))
        return max(values)

    def linearRange(self,start, end, length):
        #returns a list of a range of values over a given interval
        step=((end-start)* 1/length)
        return [start+i * step for i in xrange(length)]


    def updateQValues(self, state, action, reward):
        #updates the q table with the bellman equation
        
        if self.getEpisodesSoFar() >= self.getNumTraining():
            alpha=0
        else: 
            alpha=self.alphas[self.getEpisodesSoFar()-1] #this is where the alpha decay gets used
        
        self.q_table[(state,action)]=self.getQVal(state,action)+ alpha * (reward + self.getGamma() *(self.getMaxQValue(state) - self.getQVal(state,action)))
        
        

    # getAction
    #
    # The main method required by the game. Called every time that
    # Pacman is expected to move
    def getAction(self, state):
        #set alphas for alpha decay with the hopes to improve performance
        alpha=self.getAlpha()
        endAlpha=1.0
        length=self.getNumTraining()
        self.alphas=self.linearRange(alpha, endAlpha, length)

        #check the change in the reward from the last action
        #last_action method is taken from the banditAgents.py file from coursework 8
        new_score=state.getScore()
        old_score=self.getLastScore()
        new_reward=new_score-old_score
        if len(self.old_state)!=0:

            last_action = state.getPacmanState().configuration.direction
            last_state=self.old_state[-1]
            
            #update the Q Table
            self.updateQValues(last_state, last_action,new_reward)
        
        #select action to make
        pick=self.pick_action(state)

        #update the parameter trackers for the next game
        self.setLastScore(new_score)
        self.old_state.append(state)
        
        # We have to return an action
        return pick
            

    # Handle the end of episodes
    #
    # This is called by the game after a win or a loss.
    def final(self, state):
        #here we update Q values after each episode
        #otherwise pacman starts loving the ghosts a bit too much 
        r=state.getScore()-self.getLastScore()
        last_state=self.old_state[-1]
        last_action=state.getPacmanState().configuration.direction
        #update the Q table again to teach pacman that winning is better
        self.updateQValues(last_state,last_action,r)
        
        #reset the score and state trackers to start fresh for the next episode  
        self.old_state=[]
        self.setLastScore(0)

        
        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print '%s\n%s' % (msg,'-' * len(msg))
            self.setAlpha(0)
            self.setEpsilon(0)


