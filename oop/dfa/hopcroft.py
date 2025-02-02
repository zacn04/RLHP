from collections import deque, defaultdict

def hopcroft_minimisation(states, locations, transitions, start_state, accepting_states, max_iterations=1000):
    # initialising partitions
    non_accepting = set(states) - set(accepting_states)
    partition = [frozenset(accepting_states), frozenset(non_accepting)]

    # initialising worklist with the smaller of the two
    worklist = deque()
    if len(accepting_states) <= len(non_accepting):
        worklist.append(frozenset(accepting_states))
    else:
        worklist.append(frozenset(non_accepting))

    # refining!
    iteration = 0
    while worklist and iteration < max_iterations:
        iteration += 1
        current_split = worklist.popleft()

        # calculating pre-images
        preimage = defaultdict(set)
        for state in states:
            for transition_tuple, next_state in transitions[state].items():
                for group in partition:
                    if next_state in group:
                        preimage[group].add(state)
                        break

        # refining partitions based on these pre-images
        new_partition = []
        for group in partition:
            intersection = group & preimage[frozenset(current_split)]
            difference = group - preimage[frozenset(current_split)]

            if intersection and difference:
                new_partition.append(frozenset(intersection))
                new_partition.append(frozenset(difference))

                # update worklist
                if group in worklist:
                    worklist.remove(group)
                worklist.append(frozenset(intersection))
                worklist.append(frozenset(difference))
            else:
                new_partition.append(group)

        partition = new_partition

        if iteration >= max_iterations:
            break

    # buildiing our minimised dfa
    minimized_states = []
    minimized_transitions = {}
    state_mapping = {}

    # creating the state mapping!
    for idx, group in enumerate(partition):
        if group:
            minimized_states.append(idx)
            for state in group:
                state_mapping[state] = idx

    # build minimized transitions
    for state in minimized_states:
        minimized_transitions[state] = {}
        representative = next(iter(partition[state]))  # choosing representative state
        for transition_tuple, next_state in transitions[representative].items():
            minimized_transitions[state][transition_tuple] = state_mapping[next_state]

    # determine minimized start and accepting states
    minimized_start_state = state_mapping[start_state]
    minimized_accepting_states = {state_mapping[state] for state in accepting_states}

    return minimized_states, locations, minimized_transitions, minimized_start_state, minimized_accepting_states

def normalisation(dfa):

    states, locations, transitions, start_state, accepting_states = dfa
    location_map = {loc:i for i, loc in enumerate(locations)}
    new_locations = list(location_map.values())
    new_transitions = {}

    for state in states:
        new_transitions[state] = {}
        for transition_tuple, next_state in transitions[state].items():
            locA, locB = transition_tuple
            new_transitions[state].update({(location_map[locA], location_map[locB]) : next_state})
    return states, new_locations, new_transitions, start_state, accepting_states

if __name__ == "__main__":
    
    x = normalisation(([0, 1], [0,1,2,3], {0: {(1, 0): 1, (3, 2): 1}, 1: {(0, 1): 0, (2, 3): 0}}, 1, {0}))
    y = normalisation(([0, 1], [0,3,4,7], {0: {(4, 0): 1, (3, 7): 1}, 1: {(0, 4): 0, (7, 3): 0}}, 1, {0}))
    z = normalisation(([0, 1], [0,1,2,3], {0: {(2, 0): 1, (1, 3): 1}, 1: {(0, 2): 0, (3, 1): 0}}, 1, {0}))
    a = normalisation(([0, 1], [0,3,4,7], {0: {(4, 0): 1, (3, 7): 1}, 1: {(0, 4): 0, (7, 3): 0}}, 1, {0}))

    print(x, y, z, a, sep="\n") 