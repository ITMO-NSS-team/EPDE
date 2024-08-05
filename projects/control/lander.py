import numpy as np

import gym

if __name__ == '__main__':
    env = gym.make("LunarLander-v2",
                continuous = True,
                gravity = -10.0,
                enable_wind = True,
                wind_power = 15.0,
                turbulence_power = 1.5)

    obs_space = env.observation_space
    action_space = env.action_space
    print("The observation space: {}".format(obs_space))
    print("The action space: {}".format(action_space))

    import time 

    # Number of steps you run the agent for 
    num_steps = 15

    obs = env.reset()

    for step in range(num_steps):
        # take random action, but you can also do something more intelligent
        # action = my_intelligent_agent_fn(obs) 
        action = env.action_space.sample()
        print(action)
        print(env.step(action))
        # apply the action
        obs, reward, left_landed, right_landed, info = env.step(action)
        
        # Render the env
        env.render()

        # Wait a bit before the next frame unless you want to see a crazy fast video
        time.sleep(0.001)
        
        # If the epsiode is up, then start another one
        if left_landed and right_landed:
            env.reset()

    # Close the env
    env.close()