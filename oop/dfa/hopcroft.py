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

    return minimized_states, minimized_transitions, minimized_start_state, minimized_accepting_states
if __name__ == "__main__":
    states = [(0, 0), (0, 1), (1, 0), (1, 1)]
    locations = [0, 1, 2, 3]
    transitions = {
        (0, 0): {(0, 1): (1, 0), (3, 2): (1, 0), (4, 5): (0, 1), (7, 6): (0, 1)},
        (0, 1): {(0, 1): (1, 1), (3, 2): (1, 1), (5, 4): (0, 0), (6, 7): (0, 0)},
        (1, 0): {(1, 0): (0, 0), (2, 3): (0, 0), (4, 5): (1, 1), (7, 6): (1, 1)},
        (1, 1): {(1, 0): (0, 1), (2, 3): (0, 1), (5, 4): (1, 0), (6, 7): (1, 0)}
    }
    start_state = (0, 0)
    accepting_states = [(0, 1), (1, 0), (1, 1)]

    # Minimize the DFA using Hopcroft's algorithm
    minimized_states, minimized_transitions, minimized_start_state, minimized_accepting_states = \
        hopcroft_minimisation(states, locations, transitions, start_state, accepting_states)

    # Output the minimized DFA
    print("Minimized States:", minimized_states)
    print("Minimized Transitions:", minimized_transitions)
    print("Minimized Start State:", minimized_start_state)
    print("Minimized Accepting States:", minimized_accepting_states)