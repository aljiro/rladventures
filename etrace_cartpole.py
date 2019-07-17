import gym
import numpy as np
import matplotlib.pyplot as plt

class MChain:
    def __init__( self, size, chain = None ):
        self.size = size
        self.chain = chain

    def create( self, Obj ):
        if self.chain == None:
            return [Obj() for i in range(self.size)]
        else:
            return [self.chain.create( Obj ) for i in range(self.size)] 


class LocalDemon:
    def __init__( self ):
        self.Q = np.array([0.0, 0.0])
        self.e = np.array([0.0, 0.0])
        self.gamma = 0.9
        self.l = 0.1
        self.action = None
        self.actionType = 0

    def getMax( self ):
        return np.max(self.Q), np.argmax( self.Q )

    def act( self, env, e ):

        if np.random.rand() < e:
            Qa, self.action = self.getMax()
            self.actionType = 0
        else:
            self.action = env.action_space.sample()
            self.actionType = 1

    def ping( self, reward, demon ):
        Qa, optAction = demon.getMax()
        delta = reward + self.gamma*Qa - self.Q[self.action]
        self.e[self.action] += 1

        # if reward == 0:
        #     self.Q[self.action] = 0.0

        return delta

    def update( self, alpha, delta, reward, actionType ):    

        self.Q[0] = self.Q[0] + alpha*delta*self.e[0]
        self.Q[1] = self.Q[1] + alpha*delta*self.e[1]

        if actionType == 0:
            self.e[0] = self.gamma*self.l*self.e[0]
            self.e[1] = self.gamma*self.l*self.e[1]
        else:
           self.e = np.array([0.0, 0.0])

class GlobalDemon:

    def __init__( self, env ):
        self.xThresholds = np.array([-4.8, -0.8, 0.8])
        self.thetaThresholds = np.array([-24.0, -6.0, -1.0, 0.0, 1.0, 6.0])
        self.dxThresholds = np.array([-np.inf, -0.5, 0.5])
        self.dthetaThresholds = np.array([-np.inf, -50.0, 50.0])

        self.currentDemon = None
        self.env = env
        lc = MChain( 3, MChain( 6, MChain( 3, MChain( 3, None ))))
        self.localDemons = lc.create( LocalDemon )

    def getLocalDemon( self, state ):
        x, dx, theta, dtheta = state
        theta = theta*180.0/np.pi 
        dtheta = dtheta*180.0/np.pi

        i = np.where( self.xThresholds < x )[0][-1]
        j = np.where( self.thetaThresholds < theta )[0][-1]
        k = np.where( self.dxThresholds < dx )[0][-1]
        l = np.where( self.dthetaThresholds < dtheta )[0][-1]

        return self.localDemons[i][j][k][l]

    def act( self, state, e ):
        self.currentDemon = self.getLocalDemon( state )
        self.currentDemon.act( self.env, e )
        return self.currentDemon.action

    def update( self, alpha, state, reward ):
        newDemon = self.getLocalDemon( state )
        delta = self.currentDemon.ping( reward, newDemon )

        for i in range(3):
            for j in range(6):
                for k in range(3):
                    for l in range(3):
                        self.localDemons[i][j][k][l].update( alpha, delta, reward, self.currentDemon.actionType )




env = gym.make('CartPole-v0')
demon = GlobalDemon( env )
e = 0.1
alpha = 0.2
rewards = []
nEpisodes = 2000

for i in range(nEpisodes):
    observation = env.reset()
    total_reward = 0.0

    for t in range(200):
       #if i_episode > 690:
        #    env.render()
        #print(observation)
        action = demon.act( observation, 1.0-e )
        observation, reward, done, info = env.step( action )    
        total_reward += reward        

        if done:
            demon.update( alpha, observation, -1 )
            print("Episode finished after {} timesteps".format(t+1))
            break
        else:
            demon.update( alpha, observation, reward )

    # if i > 1200:
    #     e = 0.0
    #     alpha = 0.0
    #e = (1 - 0.001)*e

    #alpha = alpha - 0.001*alpha
    #print( "Egreedy: " + str(e) )
    rewards.append(total_reward)

plt.plot( rewards )
plt.show()
env.close()