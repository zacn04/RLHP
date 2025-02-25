from collections import deque, defaultdict

def list_hopcroft_minimisation(states, locations, transitions, start_state, accepting_states, max_iterations=1000):
    """
    Hopcroft's algorithm for DFA minimization adapted for list-based transitions.
    
    Args:
        states: List of states
        locations: List of locations (alphabet)
        transitions: Dict mapping states to lists of (loc1, loc2, next_state) tuples
        start_state: Initial state
        accepting_states: List of accepting states
        max_iterations: Max number of iterations to prevent infinite loops
        
    Returns:
        Minimized DFA components
    """
    # Converting list-based transitions to a lookup dictionary for faster processing
    trans_lookup = {}
    for state in states:
        trans_lookup[state] = {}
        for loc1, loc2, next_state in transitions.get(state, []):
            trans_lookup[state][(loc1, loc2)] = next_state
    
    # Initialize partitions
    non_accepting = set(states) - set(accepting_states)
    partition = [frozenset(accepting_states), frozenset(non_accepting)]
    
    # Initialize worklist with the smaller of the two partitions
    worklist = deque()
    if len(accepting_states) <= len(non_accepting):
        worklist.append(frozenset(accepting_states))
    else:
        worklist.append(frozenset(non_accepting))
    
    # Refining partitions
    iteration = 0
    while worklist and iteration < max_iterations:
        iteration += 1
        current_split = worklist.popleft()
        
        # Calculate pre-images
        preimage = defaultdict(set)
        for state in states:
            for loc1 in locations:
                for loc2 in locations:
                    next_state = trans_lookup[state].get((loc1, loc2))
                    if next_state is not None:
                        for group in partition:
                            if next_state in group:
                                preimage[group].add(state)
                                break
        
        # Refining partitions based on pre-images
        new_partition = []
        for group in partition:
            intersection = group & preimage[frozenset(current_split)]
            difference = group - preimage[frozenset(current_split)]
            
            if intersection and difference:
                new_partition.append(frozenset(intersection))
                new_partition.append(frozenset(difference))
                
                # Update worklist
                if group in worklist:
                    worklist.remove(group)
                    worklist.append(frozenset(intersection))
                    worklist.append(frozenset(difference))
            else:
                new_partition.append(group)
        
        partition = new_partition
        if iteration >= max_iterations:
            break
    
    # Building minimized DFA
    minimized_states = []
    minimized_transitions = {}
    state_mapping = {}
    
    # Create state mapping
    for idx, group in enumerate(partition):
        if group:
            minimized_states.append(idx)
            for state in group:
                state_mapping[state] = idx
    
    # Build minimized transitions
    for state in minimized_states:
        minimized_transitions[state] = []
        representative = next(iter(partition[state]))  # Choose a representative state
        
        for loc1 in locations:
            for loc2 in locations:
                next_state = trans_lookup[representative].get((loc1, loc2))
                if next_state is not None:
                    minimized_transitions[state].append(
                        (loc1, loc2, state_mapping[next_state])
                    )
    
    # Determine minimized start and accepting states
    minimized_start_state = state_mapping[start_state]
    minimized_accepting_states = {state_mapping[state] for state in accepting_states}
    
    return minimized_states, locations, minimized_transitions, minimized_start_state, minimized_accepting_states

def list_normalisation(dfa):
    """
    Normalize locations in a DFA with list-based transitions.
    
    Args:
        dfa: Tuple of (states, locations, transitions, start_state, accepting_states)
        
    Returns:
        Normalized DFA
    """
    states, locations, transitions, start_state, accepting_states = dfa
    
    # Create location mapping
    location_map = {loc: i for i, loc in enumerate(locations)}
    new_locations = list(location_map.values())
    
    # Update transitions with new location indices
    new_transitions = {}
    for state in states:
        new_transitions[state] = []
        for loc1, loc2, next_state in transitions.get(state, []):
            new_loc1 = location_map[loc1]
            new_loc2 = location_map[loc2]
            new_transitions[state].append((new_loc1, new_loc2, next_state))
    
    return states, new_locations, new_transitions, start_state, accepting_states