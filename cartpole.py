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
        self.gamma = 0.9
        self.action = None

    def getMax( self ):
        return np.max(self.Q), np.argmax( self.Q )

    def act( self, env, e ):
        if np.random.rand() < e:
            Qa, self.action = self.getMax()
        else:
            self.action = env.action_space.sample()
        # beta = 0.1
        # epsilon = (np.random.rand()-0.5)*beta

        # if (self.Q[0] + epsilon) < self.Q[1]:
        #     self.action = 1
        # else:
        #     self.action = 0

        return self.action

    def update( self, alpha, reward, demon ):
        Qa, action = demon.getMax()
        if reward == 1:
            self.Q[self.action] += alpha*(reward + self.gamma*Qa - self.Q[self.action])
        else:
            self.Q[self.action] = 0.0

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
        action = self.currentDemon.act( self.env, e )
        return action

    def update( self, alpha, state, reward ):
        newDemon = self.getLocalDemon( state )
        self.currentDemon.update( alpha, reward, newDemon )




env = gym.make('CartPole-v0')
demon = GlobalDemon( env )
e = 0.1
alpha = 0.2
rewards = []

for i_episode in range(1000):
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
            demon.update( alpha, observation, 0 )
            print("Episode finished after {} timesteps".format(t+1))
            break
        else:
            demon.update( alpha, observation, reward )
    
    e = e - 0.0001*e
    alpha = alpha - 0.001*alpha
    #print( "Egreedy: " + str(e) )
    rewards.append(total_reward)

plt.plot( rewards )
plt.show()
env.close()