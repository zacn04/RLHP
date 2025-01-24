from collections import deque, defaultdict

import cProfile

def hopcroft_minimisation(states, locations, transitions, start_state, accepting_states, max_iterations=1000):

    #firstly we have to initialise the partitions.
    non_accepting = set(states) - set(accepting_states)
    partition = [frozenset(accepting_states), frozenset(non_accepting)]

    print("Initial Partition:", partition)

    #then we create a worklist with the smaller of the partitions

    worklist = deque()
    if len(accepting_states) <= len(non_accepting):
        worklist.append(frozenset(accepting_states))
    else:
        worklist.append(frozenset(non_accepting))

    print("Initial Worklist:", worklist)

    # now we refine the worklist with hopcroft's algorithm. 
    iteration = 0 
    while worklist and iteration < max_iterations:

        iteration += 1
        print(f"\n--- Iteration {iteration} ---")
        print("Current Worklist:", worklist)
        print("Current Partition:", partition)

        current_split = worklist.popleft()
        print("Processing Split:", current_split)

        splits_made = False
        # for each transition tuple we find pre-images
        for loc1 in locations:
            for loc2 in locations:
                transition_tuple = (loc1, loc2)
                print(f"\nProcessing Transition Tuple: {transition_tuple}")
                preimage = defaultdict(set)
                for state in states:
                    next_state = transitions[state].get(transition_tuple, None)
                    if next_state is not None:
                        for group in partition:
                            if next_state in group:
                                preimage[group].add(state)
                                break
                print('Preimage', preimage)

            #refine groups based on pre-images
                for group in list(partition):
                        intersection = group & preimage[frozenset(current_split)]
                        difference = group - preimage[frozenset(current_split)]
                        print(f"Group: {group}, Intersection: {intersection}, Difference: {difference}")
                        if intersection and difference:
                            splits_made = True
                            partition.remove(group)
                            partition.append(frozenset(intersection))
                            partition.append(frozenset(difference))
                            print(f"Split Group: {group} into {intersection} and {difference}")
                            print("Updated Partition:", partition)

                        

                        if group in worklist:
                                worklist.remove(group)
                        if intersection:
                            worklist.append(frozenset(intersection))
                        if difference:
                            worklist.append(frozenset(difference))
                        print("Updated Worklist:", worklist)
        if not splits_made:
            #print("No further splits possible. Stopping early.")
            break

        if iteration >= max_iterations:
            #print(f"\nWarning: Maximum iteration limit ({max_iterations}) reached. Stopping early.")
            break


    # we now build the minimised dfa                            
    minimized_states = []
    minimized_transitions = {}
    state_mapping = {}


    #new state IDs! yay 

    for idx, group in enumerate(partition):
        if group:
            minimized_states.append(idx)
            for state in group:
                state_mapping[state] = idx
    print("\nState Mapping:", state_mapping)

    #transitions are now minimised 

    for state in minimized_states:
        minimized_transitions[state] = {}
        for loc1 in locations:
            for loc2 in locations:
                transition_tuple = (loc1, loc2)
                # find a representative state from the group
                representative = next(iter(partition[state]))
                next_state = transitions[representative].get(transition_tuple, None)
                if next_state is not None:
                    minimized_transitions[state][transition_tuple] = state_mapping[next_state]

    print("Minimized Transitions:", minimized_transitions)

    minimized_start_state = state_mapping[start_state]
    minimized_accepting_states = set()
    for state in accepting_states:
        minimized_accepting_states.add(state_mapping[state])

    print("Minimized Start State:", minimized_start_state)
    print("Minimized Accepting States:", minimized_accepting_states)
    
    return minimized_states, minimized_transitions, minimized_start_state, minimized_accepting_states


## TEST ## 

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

