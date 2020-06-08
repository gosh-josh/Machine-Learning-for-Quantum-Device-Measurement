import sys
import matplotlib.pyplot as plt
from offline_test_environment_creation import double_dot_2d

sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../environments')
sys.path.append('../utilities')
sys.path.append('/environments')
sys.path.append('../utilities')

from datetime import datetime
import numpy as np

def offline_train_play_episode(block_size,env_list, total_t, n_episode, experience_replay_buffer, model, gamma,
                       batch_size, epsilon, epsilon_change, epsilon_min, summary_writer, MaxStep=100):
    t0 = datetime.now()
    env_id = np.random.randint(0, len(env_list))

    env = double_dot_2d(block_size , env_list[env_id])

    loss = None

    total_time_training = 0
    num_steps_in_episode = 0
    episode_reward = 0

    location = env.get_location()

    statistics = env.get_statistics(location)

    done = False
    while not done:
        if num_steps_in_episode > MaxStep:
            break

        model.count_exp += 1
        # Take action
        action = model.sample_action(env,statistics,location,epsilon)
        prev_state = statistics
        prev_loc_state = location
        # prev_obs=obs
        statistics, reward, done, location, revisited = env.step(action)
        while revisited == True:
            action = model.sample_random_action(env, statistics, location)
            statistics, reward, done, location, revisited = env.step(action)

        episode_reward += reward
        episode_reward += -1

        if num_steps_in_episode == MaxStep and done is False:
            reward = -4
            done = True

        # update the model

        neighborMaps = env.get_neighborMap(prev_loc_state)
        neighborMaps_next = env.get_neighborMap(location)

        model.add_experience(prev_state, prev_loc_state, action, reward,
                             statistics, location, done, neighborMaps, neighborMaps_next, env_id)

        # Train the model, keep track of time
        t0_2 = datetime.now()
        summary, loss = model.fit_prioritized_exp_replay(env_list)

        # loss = learn(model, target_model, experience_replay_buffer, gamma, batch_size)
        dt = datetime.now() - t0_2

        total_time_training += dt.total_seconds()
        num_steps_in_episode += 1

        # state = next_state
        total_t += 1

        epsilon = max(epsilon - epsilon_change, epsilon_min)

        # summary_str = sess.run(merged_summary_op, feed_dict={x: batch_xs, y: batch_ys})
    if summary is not None:
        summary_writer.add_summary(summary, n_episode)

    episode_reward = float("{0:.2f}".format(episode_reward))

    return total_t, episode_reward, (datetime.now() - t0), num_steps_in_episode, \
           total_time_training / num_steps_in_episode, epsilon, loss, summary_writer

def burn_in_experience_offline(env_list,block_size, experience_replay_buffer, model, MaxStep=100):
    epsilon = 1

    env_id = np.random.randint(0, len(env_list))

    env = double_dot_2d(block_size , env_list[env_id])

    num_steps_in_episode = 0
    episode_reward = 0

    state_list = []
    prev_state_list = []
    prev_loc_state_list = []
    loc_state_list = []
    action_list = []
    reward_list = []
    done_list = []
    neighborMaps_list = []
    neighborMaps_next_list = []
    env_id_list = []
    env_id_list.append(env_id)

    loc_state = env.get_location()

    state = env.get_statistics(loc_state)

    done = False
    terminate = False
    while not done:
        if num_steps_in_episode > MaxStep:
            break

        model.count_exp += 1
        # Take action
        action = model.sample_action(env, state, loc_state, epsilon)

        prev_state = state
        prev_loc_state = loc_state
        # prev_obs=obs
        state, reward, done, loc_state, revisited = env.step(action)  # last output is the location of the state

        while revisited == True:
            action = model.sample_random_action(env, state, loc_state)
            state, reward, done, loc_state, revisited = env.step(action)

        episode_reward += reward
        episode_reward += -1

        if num_steps_in_episode == MaxStep and done is False:
            reward = -4
            done = True
            terminate = True

        # update the model

        neighborMaps = env.get_neighborMap(prev_loc_state)
        neighborMaps_next = env.get_neighborMap(loc_state)

        env_id_list.append(env_id)
        neighborMaps_next_list.append(neighborMaps_next)
        neighborMaps_list.append(neighborMaps)
        done_list.append(done)
        reward_list.append(reward)
        action_list.append(action)
        loc_state_list.append(loc_state)
        prev_loc_state_list.append(prev_loc_state)
        state_list.append(state)
        prev_state_list.append(prev_state)

        num_steps_in_episode += 1

        # state = next_state

    if terminate is False:  # found the Target
        model.add_experience(prev_state_list, prev_loc_state_list, action_list, reward_list,
                             state_list, loc_state_list, done_list, neighborMaps_list, neighborMaps_next_list,
                             env_id_list)
        return True
    else:
        return False