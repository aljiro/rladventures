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
        self.gamma = 1.0
        self.action = None

    def getMax( self ):
        return np.max(self.Q), np.argmax( self.Q )

    def act( self, env, e ):
        if np.random.random_sample() < e:
            self.action = env.action_space.sample()
        else:
            _, self.action = self.getMax()

    def update( self, alpha, reward, demon ):
        Qa, _ = demon.getMax()
        if reward == 1:
            self.Q[self.action] += alpha*(reward + self.gamma*Qa - self.Q[self.action])
        else:
            self.Q[self.action] = 0.0


class GlobalDemon:

    def __init__( self, env ):        

        self.currentDemon = None
        self.env = env
        lc = MChain( 3, MChain( 6, MChain( 3, MChain( 3, None ))))
        self.localDemons = lc.create( LocalDemon )


    def getLocalDemon( self, state ):
        xThresholds = np.array([-4.8, -0.8, 0.8])
        thetaThresholds = np.array([-24.0, -6.0, -2.0, 0.0, 2.0, 6.0])
        dxThresholds = np.array([-np.inf, -0.5, 0.5])
        dthetaThresholds = np.array([-np.inf, -50.0, 50.0])

        x, dx, theta, dtheta = state
        theta = theta*180.0/np.pi 
        dtheta = dtheta*180.0/np.pi

        i = np.where( xThresholds < x )[0][-1]
        j = np.where( thetaThresholds < theta )[0][-1]
        k = np.where( dxThresholds < dx )[0][-1]
        l = np.where( dthetaThresholds < dtheta )[0][-1]

        return self.localDemons[i][j][k][l]

    def act( self, state, e ):
        self.currentDemon = self.getLocalDemon( state )
        self.currentDemon.act( self.env, e )
        return self.currentDemon.action

    def update( self, alpha, state, reward ):
        newDemon = self.getLocalDemon( state )
        self.currentDemon.update( alpha, reward, newDemon )


class Task:

    def __init__(self, nEpisodes, T ):
        self.decay = 35.0
        self.min_e = 0.01
        self.min_alpha = 0.15
        self.nEpisodes = nEpisodes
        self.T = T

    def log( self, state, i ):
        if self.X.size == 0:
            self.X = np.array([state])
        else:
            self.X = np.concatenate( (self.X, [state]), axis=0 )
            self.episodes = np.append( self.episodes, i )

    def plot( self ):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot( self.rewards )

        (m,n) = self.X.shape
        Y = self.X
        Y = Y - Y.mean( axis=1, keepdims=True )
        C = Y.T.dot(Y)/float(m)
        w,v = np.linalg.eig( C )
        z = Y.dot(v.T)
        idxLow = np.array(self.rewards) < 40
        idxHigh = np.array(self.rewards) > 150

        for i in range(idxLow.size):
            idxs = 
        ax2.plot( z[idxLow,-1], z[idxLow,-2], 'r' )
        ax2.plot( z[idxHigh,-1], z[idxHigh,-2], 'b' )
        plt.show()


    def run( self ):

        self.rewards = []
        self.X = np.array([])
        self.episodes = np.array([])
        e = 1.
        alpha = 1.
        env = gym.make('CartPole-v1')
        demon = GlobalDemon( env )        

        f = lambda x, min_x : max( min_x, min( 1.0, 1.0 - np.log10((x + 1)/self.decay)) )

        for i in range(self.nEpisodes):
            observation = env.reset()
            total_reward = 0.0

            for t in range(self.T):
                #if i_episode > 690:
                   #env.render()
                #print(observation)
                action = demon.act( observation, e )
                observation, reward, done, info = env.step( action ) 
                total_reward += reward        

                self.log( observation, i )

                if done:
                    demon.update( alpha, observation, 0 )
                    print("Episode finished after {} timesteps".format(t+1))
                    break
                else:
                    demon.update( alpha, observation, reward )

            
            e = f( i, self.min_e )
            alpha = f( i, self.min_alpha )   
            self.rewards.append(total_reward)

        env.close()

if __name__ == "__main__":
    task = Task( 500, 200 )
    task.run()
    task.plot()