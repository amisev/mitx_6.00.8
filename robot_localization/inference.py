#!/usr/bin/env python
# inference.py
# Base code by George H. Chen (georgehc@mit.edu) -- updated 10/18/2016
import collections
import sys

import graphics
import numpy as np
import robot


# Throughout the code, we use these variables.
# Do NOT change these (but you'll need to use them!):
# - all_possible_hidden_states: a list of possible hidden states
# - all_possible_observed_states: a list of possible observed states
# - prior_distribution: a distribution over states
# - transition_model: a function that takes a hidden state and returns a
#     Distribution for the next state
# - observation_model: a function that takes a hidden state and returns a
#     Distribution for the observation from that hidden state
all_possible_hidden_states = robot.get_all_hidden_states()
all_possible_observed_states = robot.get_all_observed_states()
prior_distribution = robot.initial_distribution()
transition_model = robot.transition_model
observation_model = robot.observation_model


# You may find this function helpful for computing logs without yielding a
# NumPy warning when taking the log of 0.
def careful_log(x):
    # computes the log of a non-negative real number
    if x == 0:
        return -np.inf
    else:
        return np.log(x)


# -----------------------------------------------------------------------------
# Functions for you to implement
#

def forward_backward(observations):
    """
    Input
    -----
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    robot.py and see how it is used in both robot.py and the function
    generate_data() above, and the i-th Distribution should correspond to time
    step i
    """

    # -------------------------------------------------------------------------
    # YOUR CODE GOES HERE
    #

    # define transition matrix
    A = np.zeros((len(all_possible_hidden_states), len(all_possible_hidden_states)))
    for index, current_state in enumerate(all_possible_hidden_states):
        for next_state, next_state_probability in dict(robot.transition_model(current_state)).items():
            A[index, all_possible_hidden_states.index(next_state)] = next_state_probability
    # define observation matrix
    B = np.zeros((len(all_possible_hidden_states), len(all_possible_observed_states)))
    for index, current_state in enumerate(all_possible_hidden_states):
        for next_state, next_state_probability in dict(robot.observation_model(current_state)).items():
            B[index, all_possible_observed_states.index(next_state)] = next_state_probability

    # list(map(lambda x_i: None if x_i is None else all_possible_observed_states.index(x_i) , x))

    num_time_steps = len(observations)
    forward_messages = [None] * num_time_steps
    forward_messages[0] = [prior_distribution]
    # Compute the forward messages
    for i, o in enumerate(observations[:-1]):
            if o is None:
                fm = robot.Distribution(dict(zip(all_possible_hidden_states, np.sum([forward_messages[i][0][state]*1*A[j, :].T for j, state in enumerate(all_possible_hidden_states)], axis=0))))
            else:
                fm = robot.Distribution(dict(zip(all_possible_hidden_states, np.sum([forward_messages[i][0][state]*B[j, all_possible_observed_states.index(o)]*A[j, :].T for j, state in enumerate(all_possible_hidden_states)], axis=0))))
            fm.renormalize()
            forward_messages[i+1] = [fm]

    backward_messages = [None] * num_time_steps
    # Compute the backward messages
    # first backward message
    # backward messages are indexed from 0 to num_time_steps-1
    for i, o in enumerate(observations[::-1][:1]):
            if o is None:
                fm = robot.Distribution(dict(zip(all_possible_hidden_states, np.sum([A[:, j] for j, state in enumerate(all_possible_hidden_states)], axis=0))))
            else:
                fm = robot.Distribution(dict(zip(all_possible_hidden_states, np.sum([B[j, all_possible_observed_states.index(o)]*A[:, j] for j, state in enumerate(all_possible_hidden_states)], axis=0))))
            fm.renormalize()
            backward_messages[num_time_steps - i - 1] = [fm]
    # next backward messages
    for i, o in enumerate(observations[::-1][1:-1]):
            if o is None:
                fm = robot.Distribution(dict(zip(all_possible_hidden_states, np.sum([backward_messages[num_time_steps - i - 1][0][state]*1*A[:, j] for j, state in enumerate(all_possible_hidden_states)], axis=0))))
            else:
                fm = robot.Distribution(dict(zip(all_possible_hidden_states, np.sum([backward_messages[num_time_steps - i - 1][0][state]*B[j, all_possible_observed_states.index(o)]*A[:, j] for j, state in enumerate(all_possible_hidden_states)], axis=0))))
            fm.renormalize()
            backward_messages[num_time_steps - i - 2] = [fm]

    marginals = [None] * num_time_steps  # remove this
    # Compute the marginals
    for i, o in enumerate(observations[:-1]):
            if o is None:
                fm = robot.Distribution(dict(zip(all_possible_hidden_states, [forward_messages[i][0][state]*backward_messages[i+1][0][state]*1 for j, state in enumerate(all_possible_hidden_states)])))
            else:
                fm = robot.Distribution(dict(zip(all_possible_hidden_states, [forward_messages[i][0][state]*backward_messages[i+1][0][state]*B[j, all_possible_observed_states.index(o)] for j, state in enumerate(all_possible_hidden_states)])))
            fm.renormalize()
            marginals[i] = fm
    # in last marginal we exclude backward part
    for i, o in enumerate(observations[-1:]):
            if o is None:
                fm = robot.Distribution(dict(zip(all_possible_hidden_states, [forward_messages[num_time_steps - 1][0][state]*1 for j, state in enumerate(all_possible_hidden_states)])))
            else:
                fm = robot.Distribution(dict(zip(all_possible_hidden_states, [forward_messages[num_time_steps - 1][0][state]*B[j, all_possible_observed_states.index(o)] for j, state in enumerate(all_possible_hidden_states)])))
            fm.renormalize()
            marginals[num_time_steps - 1] = fm

    return marginals


def Viterbi(observations):
    """
    Input
    -----
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of esimated hidden states, each encoded as a tuple
    (<x>, <y>, <action>)
    """

    # -------------------------------------------------------------------------
    # YOUR CODE GOES HERE
    #
    # define transition matrix
    A = np.zeros((len(all_possible_hidden_states), len(all_possible_hidden_states)))
    for index, current_state in enumerate(all_possible_hidden_states):
        for next_state, next_state_probability in dict(robot.transition_model(current_state)).items():
            A[index, all_possible_hidden_states.index(next_state)] = next_state_probability
    # define observation matrix
    B = np.zeros((len(all_possible_hidden_states), len(all_possible_observed_states)))
    for index, current_state in enumerate(all_possible_hidden_states):
        for next_state, next_state_probability in dict(robot.observation_model(current_state)).items():
            B[index, all_possible_observed_states.index(next_state)] = next_state_probability

    num_time_steps = len(observations)
    # define node potentials
    phi = np.zeros((len(all_possible_hidden_states), num_time_steps))
    n_prior_distribution = np.array([prior_distribution[state] for state in all_possible_hidden_states])
    for i, o in enumerate(observations[:1]):
        if o is None:
            phi[:, i] = -1*np.log2(n_prior_distribution[:, None].T)
        else:
            phi[:, i] = -1*np.log2(np.multiply(n_prior_distribution, B[:, all_possible_observed_states.index(o)])[:, None].T)
    for i, o in enumerate(observations[1:]):
        if o is None:
            phi[:, i + 1] = -1*np.log2(np.ones(phi.shape[0])[:, None].T)
        else:
            phi[:, i + 1] = -1*np.log2(B[:, all_possible_observed_states.index(o)])

    psi = -1*np.log2(A)

    messages = np.zeros((len(all_possible_hidden_states), num_time_steps - 1))
    traceback_messages = np.zeros((len(all_possible_hidden_states), num_time_steps - 1))

    messages[:, 0] = [np.min(phi[:, 0] + psi[:, j]) for j, state in enumerate(all_possible_hidden_states)]
    traceback_messages[:, 0] = [np.argmin(phi[:, 0] + psi[:, j]) for j, state in enumerate(all_possible_hidden_states)]
    for time_stamp in np.arange(num_time_steps - 2):
        messages[:, time_stamp + 1] = [np.min(phi[:, time_stamp + 1] + psi[:, j] + messages[:, time_stamp]) for j, state in enumerate(all_possible_hidden_states)]
        traceback_messages[:, time_stamp + 1] = [np.argmin(phi[:, time_stamp + 1] + psi[:, j] + messages[:, time_stamp]) for j, state in enumerate(all_possible_hidden_states)]

    last_state = np.argmin(phi[:, -1] + messages[:, -1])

    estimated_hidden_states = [None] * num_time_steps
    estimated_hidden_states[num_time_steps-1] = all_possible_hidden_states[last_state]

    for time_stamp in np.arange(num_time_steps - 1, 0, -1):
        estimated_hidden_states[time_stamp - 1] = all_possible_hidden_states[int(traceback_messages[all_possible_hidden_states.index(estimated_hidden_states[time_stamp]), time_stamp - 1])]

    return estimated_hidden_states


def second_best(observations):
    """
    Input
    -----
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of esimated hidden states, each encoded as a tuple
    (<x>, <y>, <action>)
    """

    #
    # serial LVA implementaition
    #

    # define transition matrix
    A = np.zeros((len(all_possible_hidden_states), len(all_possible_hidden_states)))
    for index, current_state in enumerate(all_possible_hidden_states):
        for next_state, next_state_probability in dict(robot.transition_model(current_state)).items():
            A[index, all_possible_hidden_states.index(next_state)] = next_state_probability
    # define observation matrix
    B = np.zeros((len(all_possible_hidden_states), len(all_possible_observed_states)))
    for index, current_state in enumerate(all_possible_hidden_states):
        for next_state, next_state_probability in dict(robot.observation_model(current_state)).items():
            B[index, all_possible_observed_states.index(next_state)] = next_state_probability

    num_time_steps = len(observations)
    # define node potentials
    phi = np.zeros((len(all_possible_hidden_states), num_time_steps))
    n_prior_distribution = np.array([prior_distribution[state] for state in all_possible_hidden_states])
    for i, o in enumerate(observations[:1]):
        if o is None:
            phi[:, i] = -1*np.log2(n_prior_distribution[:, None].T)
        else:
            phi[:, i] = -1*np.log2(np.multiply(n_prior_distribution, B[:, all_possible_observed_states.index(o)])[:, None].T)
    for i, o in enumerate(observations[1:]):
        if o is None:
            phi[:, i + 1] = -1*np.log2(np.ones(phi.shape[0])[:, None].T)
        else:
            phi[:, i + 1] = -1*np.log2(B[:, all_possible_observed_states.index(o)])

    psi = -1*np.log2(A)

    # k most likely paths
    k = 2
    # here we find first best path (global)
    messages = np.zeros((len(all_possible_hidden_states), num_time_steps - 1))
    traceback_messages = np.zeros((len(all_possible_hidden_states), num_time_steps - 1))
    #
    path = np.zeros((len(all_possible_hidden_states), num_time_steps - 1, k))
    #
    messages[:, 0] = [np.min(phi[:, 0] + psi[:, j]) for j, state in enumerate(all_possible_hidden_states)]
    traceback_messages[:, 0] = [np.argmin(phi[:, 0] + psi[:, j]) for j, state in enumerate(all_possible_hidden_states)]
    path[:, 0, 0] = messages[:, 0]
    for time_stamp in np.arange(0, num_time_steps - 2):
        messages[:, time_stamp + 1] = [np.min(phi[:, time_stamp + 1] + psi[:, j] + messages[:, time_stamp]) for j, state in enumerate(all_possible_hidden_states)]
        traceback_messages[:, time_stamp + 1] = [np.argmin(phi[:, time_stamp + 1] + psi[:, j] + messages[:, time_stamp]) for j, state in enumerate(all_possible_hidden_states)]
        path[:, time_stamp + 1, 0] = messages[:, time_stamp + 1]

    last_state = np.argmin(phi[:, -1] + messages[:, -1])
    estimated_hidden_states_1 = [None] * num_time_steps

    estimated_hidden_states_1[num_time_steps-1] = all_possible_hidden_states[last_state]

    for time_stamp in np.arange(num_time_steps - 1, 0, -1):
        estimated_hidden_states_1[time_stamp - 1] = all_possible_hidden_states[int(traceback_messages[all_possible_hidden_states.index(estimated_hidden_states_1[time_stamp]), time_stamp - 1])]

    # here we find second best path
    # time, then second best path will merge with first best path
    time_merge_happend = 0
    # state, predecessor of current state
    state_merge_happend = all_possible_hidden_states[0]
    # minimal cost of second best path
    second_best_path_candidate_min = np.inf
    # last state of first best path, to make comparisons with second best path
    best_state_last = all_possible_hidden_states.index(estimated_hidden_states_1[-1])
    # initialize second best path
    path[:, 0, 1] = path[:, 0, 0]
    for time_stamp in np.arange(num_time_steps - 2):
        # here we hold t + 1, t best states
        # best_state_next = all_possible_hidden_states.index(estimated_hidden_states_1[time_stamp + 2])
        best_state_current = all_possible_hidden_states.index(estimated_hidden_states_1[time_stamp + 1])
        best_state_previous = all_possible_hidden_states.index(estimated_hidden_states_1[time_stamp + 0])
        #
        second_best_path_current = phi[best_state_previous, time_stamp] + psi[best_state_previous, best_state_current] + path[best_state_previous, time_stamp, 1]
        second_best_path_candidate = phi[:, time_stamp + 1] + psi[:, best_state_current] + path[:, time_stamp, 0]
        # compare second best path candidate cost with first best path cost
        # (from begining to current state)
        if second_best_path_candidate[second_best_path_candidate.argsort()[:2][1]] < second_best_path_current:
            #
            path[best_state_current, time_stamp + 1, 1] = second_best_path_candidate[second_best_path_candidate.argsort()[:2][1]]
            #
            if second_best_path_candidate[second_best_path_candidate.argsort()[:2][1]] + path[best_state_last, -1, 0] - path[best_state_current, time_stamp, 0] < second_best_path_candidate_min:
                second_best_path_candidate_min = second_best_path_candidate[second_best_path_candidate.argsort()[:2][1]] + path[best_state_last, -1, 0] - path[best_state_current, time_stamp, 0]
                # store time, then merge happend
                time_merge_happend = time_stamp + 1
                # store state, with second best path
                state_merge_happend = second_best_path_candidate.argsort()[:2][1]
        else:
            path[best_state_current, time_stamp + 1, 1] = second_best_path_current

    # print('time merged happend: ', time_merge_happend)
    estimated_hidden_states = [None] * num_time_steps
    estimated_hidden_states[time_merge_happend:] = estimated_hidden_states_1[time_merge_happend:]
    estimated_hidden_states[time_merge_happend - 1] = all_possible_hidden_states[state_merge_happend]
    for time_stamp in np.arange(time_merge_happend - 1, 0, -1):
        estimated_hidden_states[time_stamp - 1] = all_possible_hidden_states[int(traceback_messages[all_possible_hidden_states.index(estimated_hidden_states[time_stamp]), time_stamp - 1])]

    return estimated_hidden_states
# -----------------------------------------------------------------------------
# Generating data from the hidden Markov model
#


def generate_data(num_time_steps, make_some_observations_missing=False,
                  random_seed=None):
    # generate samples from this project's hidden Markov model
    hidden_states = []
    observations = []

    # if the random seed is not None, then this makes the randomness
    # deterministic, which may be helpful for debug purposes
    np.random.seed(random_seed)

    # draw initial state and emit an observation
    initial_state = prior_distribution.sample()
    initial_observation = observation_model(initial_state).sample()

    hidden_states.append(initial_state)
    observations.append(initial_observation)

    for time_step in range(1, num_time_steps):
        # move the robot
        prev_state = hidden_states[-1]
        new_state = transition_model(prev_state).sample()

        # maybe emit an observation
        if not make_some_observations_missing:
            new_observation = observation_model(new_state).sample()
        else:
            if np.random.rand() < .1:  # 0.1 prob. of observation being missing
                new_observation = None
            else:
                new_observation = observation_model(new_state).sample()

        hidden_states.append(new_state)
        observations.append(new_observation)

    return hidden_states, observations


# -----------------------------------------------------------------------------
# Main
#

def main():
    # flags
    make_some_observations_missing = False
    use_graphics = True
    need_to_generate_data = True

    # parse command line arguments
    for arg in sys.argv[1:]:
        if arg == '--missing':
            make_some_observations_missing = True
        elif arg == '--nographics':
            use_graphics = False
        elif arg.startswith('--load='):
            filename = arg[7:]
            hidden_states, observations = robot.load_data(filename)
            need_to_generate_data = False
            num_time_steps = len(hidden_states)

    # if no data is loaded, then generate new data
    if need_to_generate_data:
        num_time_steps = 100
        hidden_states, observations = \
            generate_data(num_time_steps,
                          make_some_observations_missing)

    print('Running forward-backward...')
    marginals = forward_backward(observations)
    print("\n")

    timestep = 2
    print("Most likely parts of marginal at time %d:" % (timestep))
    if marginals[timestep] is not None:
        print(sorted(marginals[timestep].items(),
                     key=lambda x: x[1],
                     reverse=True)[:10])
    else:
        print('*No marginal computed*')
    print("\n")

    print('Running Viterbi...')
    estimated_states = Viterbi(observations)
    print("\n")

    print("Last 10 hidden states in the MAP estimate:")
    for time_step in range(num_time_steps - 10 - 1, num_time_steps):
        if estimated_states[time_step] is None:
            print('Missing')
        else:
            print(estimated_states[time_step])
    print("\n")

    print('Finding second-best MAP estimate...')
    estimated_states2 = second_best(observations)
    print("\n")

    print("Last 10 hidden states in the second-best MAP estimate:")
    for time_step in range(num_time_steps - 10 - 1, num_time_steps):
        if estimated_states2[time_step] is None:
            print('Missing')
        else:
            print(estimated_states2[time_step])
    print("\n")

    difference = 0
    difference_time_steps = []
    for time_step in range(num_time_steps):
        if estimated_states[time_step] != hidden_states[time_step]:
            difference += 1
            difference_time_steps.append(time_step)
    print("Number of differences between MAP estimate and true hidden " +
          "states:", difference)
    if difference > 0:
        print("Differences are at the following time steps: " +
              ", ".join(["%d" % time_step
                         for time_step in difference_time_steps]))
    print("\n")

    difference = 0
    difference_time_steps = []
    for time_step in range(num_time_steps):
        if estimated_states2[time_step] != hidden_states[time_step]:
            difference += 1
            difference_time_steps.append(time_step)
    print("Number of differences between second-best MAP estimate and " +
          "true hidden states:", difference)
    if difference > 0:
        print("Differences are at the following time steps: " +
              ", ".join(["%d" % time_step
                         for time_step in difference_time_steps]))
    print("\n")

    difference = 0
    difference_time_steps = []
    for time_step in range(num_time_steps):
        if estimated_states[time_step] != estimated_states2[time_step]:
            difference += 1
            difference_time_steps.append(time_step)
    print("Number of differences between MAP and second-best MAP " +
          "estimates:", difference)
    if difference > 0:
        print("Differences are at the following time steps: " +
              ", ".join(["%d" % time_step
                         for time_step in difference_time_steps]))
    print("\n")

    # display
    if use_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()


if __name__ == '__main__':
    main()
