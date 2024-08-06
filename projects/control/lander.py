import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import time

import gym
import scipy.special

if __name__ == '__main__':
    env = gym.make("LunarLander-v2",
                continuous = True,
                gravity = -9.8,
                enable_wind = True,
                wind_power = 1.0,
                turbulence_power = 1.5,
                render_mode="rgb_array")

    obs_space = env.observation_space
    action_space = env.action_space
    print("The observation space: {}".format(obs_space))
    print("The action space: {}".format(action_space))

    import time 

    # Number of steps you run the agent for 
    num_steps = 200

    k = 3
    main_thrust = lambda x: (np.power(x, k/2.-1)*np.exp(-x/2.))/(2**(k/2.)*scipy.special.gamma(k/2.))+0.5
    test_range = np.linspace(0, 5, 100)
    plt.plot(test_range, main_thrust(test_range))
    plt.show()

    obs = env.reset()
    observations = [obs[0],]
    print(f'Initial state is {observations}')
    acts = [[0, 0],]
    left_landed = False; right_landed = False
    for step in range(num_steps):
        # take random action, but you can also do something more intelligent
        # action = my_intelligent_agent_fn(obs) 
        if not (left_landed or right_landed):
            print(observations[-1][3])
            hor_thrust = np.sign(observations[-1][3]) * (np.abs(observations[-1][3])+0.2)
            action = [main_thrust(observations[-1][1]), hor_thrust]#env.action_space.sample()
        else:
            action = [0., 0.]

        acts.append(action)
        # print(action)
        # print(env.step(action))
        # apply the action
        obs, reward, left_landed, right_landed, info = env.step(action)
        observations.append(obs)
        # Render the env
        # Wait a bit before the next frame unless you want to see a crazy fast video
        # time.sleep(0.001)
        
        # If the epsiode is up, then start another one
        if left_landed or right_landed:
            print(f'Ouch: l {left_landed} r {right_landed} on {step}')
        if left_landed or right_landed:
            print('Touched!')
            env.reset()
            break

    # Close the env
    env.close()
    # print(len(observations))
    observations = np.stack(observations, axis = 1)
    acts = np.array(acts).T
    acts[0, :] = np.where(acts[0, :] > 0.5, acts[0, :], 0.)    
    acts[1, :] = np.where(np.abs(acts[1, :]) < -0.5, 0., acts[1, :])    
    # print('Obs:', observations.shape, ' Acts:', acts.shape)
    # print(acts[0, :])
    # print(acts[1, :])

    

    # print([len(obs) for obs in observations])

    x    = observations[0, :].reshape(-1)
    y    = observations[1, :].reshape(-1)
    cols = np.linspace(0,1, x.size)


    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    fig, axs = plt.subplots() # 1, 1, sharex=True, sharey=True

    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(cols.min(), cols.max())
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    # Set the values used for colormapping
    lc.set_array(cols)
    lc.set_linewidth(2)
    line = axs.add_collection(lc)
    fig.colorbar(line, ax=axs)

    eps = 0.1
    axs.set_xlim(-max(np.abs(x.min()), np.abs(x.max()))- eps, max(np.abs(x.min()), np.abs(x.max()))+eps)
    axs.set_ylim(y.min()-eps, y.max()+eps)

    every = int(observations[0, :].size / 10.)
    origins = np.stack([observations[0, :].reshape(-1)[::every], observations[1, :].reshape(-1)[::every]], axis = 0)

    plt.quiver(*origins, np.sin(observations[4, :][::every]), np.cos(observations[4, :][::every]), color='r', scale=10)
    plt.quiver(*origins, observations[2, :][::every], observations[3, :][::every], color='b', scale=5)

    plt.show()

    plt.plot(np.arange(observations[2, :].size), observations[2, :], color = 'k')
    plt.plot(np.arange(observations[3, :].size), observations[3, :], color = 'r')
    plt.show()

    plt.plot(np.arange(acts[0, :].size), acts[0, :], color = 'k')
    plt.plot(np.arange(acts[1, :].size), acts[1, :], color = 'r')
    plt.show()