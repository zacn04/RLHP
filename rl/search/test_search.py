from oop.gadgets.gadgetdefs import (
    AntiParallel2Toggle, 
    Crossing2Toggle, 
    Toggle2,
    AntiParallelLocking2Toggle,
    CrossingLocking2Toggle,
    ParallelLocking2Toggle,
    Door,
    SelfClosingDoor
)
from oop.gadgets.gadgetlike import GadgetNetwork
from rl.search.exhaustive.search import find_simulation, SearchStrategy, print_gadget_definition, format_operation

def test_ap2t_sim_c2t():
    """Test if search can find that two AP2Ts can simulate C2T"""
    print("\nTesting AP2T -> C2T simulation search...")
    
    # Initial gadgets
    ap2t1 = AntiParallel2Toggle()
    ap2t2 = AntiParallel2Toggle()
    ap2t2.setCurrentState(1)
    
    # Target gadget
    target = Crossing2Toggle()
    
    # Run search
    path = find_simulation(
        initial_gadgets=[ap2t1, ap2t2],
        target_gadget=target,
        strategy=SearchStrategy.BFS,
        max_depth=4,
        verbose=True
    )
    
    if path:
        print("Found solution!")
        print("Operations:")
        for op in path:
            print(f"  {format_operation(op)}")
        return True
    else:
        print("No solution found")
        return False

def test_cl2t_sim_pl2t():
    """Test if search can find that two CL2Ts can simulate PL2T"""
    print("\nTesting CL2T -> PL2T simulation search...")
    
    # Initial gadgets
    cl2t1 = CrossingLocking2Toggle()
    cl2t2 = CrossingLocking2Toggle()
    
    # Target gadget
    target = ParallelLocking2Toggle()
    
    # Run search
    path = find_simulation(
        initial_gadgets=[cl2t1, cl2t2],
        target_gadget=target,
        strategy=SearchStrategy.BFS,
        max_depth=4,
        verbose=True
    )
    
    if path:
        print("Found solution!")
        print("Operations:")
        for op in path:
            print(f"  {format_operation(op)}")
        return True
    else:
        print("No solution found")
        return False

def test_c2t_sim_p2t():
    """Test if search can find that two C2Ts can simulate Toggle2"""
    print("\nTesting C2T -> Toggle2 simulation search...")
    
    # Initial gadgets
    c2t1 = Crossing2Toggle()
    c2t2 = Crossing2Toggle()
    
    # Target gadget
    target = Toggle2()
    
    # Run search
    path = find_simulation(
        initial_gadgets=[c2t1, c2t2],
        target_gadget=target,
        strategy=SearchStrategy.BFS,
        max_depth=4,
        verbose=True
    )
    
    if path:
        print("Found solution!")
        print("Operations:")
        for op in path:
            print(f"  {format_operation(op)}")
        return True
    else:
        print("No solution found")
        return False

def test_door_sim_scd():
    """Test if two Doors can simulate a Self-Closing Door"""
    print("\nTesting Door -> SelfClosingDoor simulation search...")
    
    # Initial gadgets
    door1 = Door()
    door2 = Door()
    
    # Target gadget
    target = SelfClosingDoor()
    
    # Run search
    path = find_simulation(
        initial_gadgets=[door1, door2],
        target_gadget=target,
        strategy=SearchStrategy.BFS,
        max_depth=4,
        verbose=True
    )
    
    if path:
        print("Found solution!")
        print("Operations:")
        for op in path:
            print(f"  {format_operation(op)}")
        return True
    else:
        print("No solution found")
        return False

def test_apl2t_sim_cl2t():
    """Test if two APL2Ts can simulate a CL2T"""
    print("\nTesting APL2T -> CL2T simulation search...")
    
    # Initial gadgets
    apl2t1 = AntiParallelLocking2Toggle()
    apl2t2 = AntiParallelLocking2Toggle()
    
    # Target gadget
    target = CrossingLocking2Toggle()
    
    # Run search
    path = find_simulation(
        initial_gadgets=[apl2t1, apl2t2],
        target_gadget=target,
        strategy=SearchStrategy.BFS,
        max_depth=4,
        verbose=True
    )
    
    if path:
        print("Found solution!")
        print("Operations:")
        for op in path:
            print(f"  {format_operation(op)}")
        return True
    else:
        print("No solution found")
        return False

def test_complex_simulation():
    """Test if combination of different gadget types can simulate another"""
    print("\nTesting complex gadget simulation...")
    
    # Initial gadgets
    ap2t = AntiParallel2Toggle()
    cl2t = CrossingLocking2Toggle()
    
    # Target gadget
    target = ParallelLocking2Toggle()
    
    # Run search
    path = find_simulation(
        initial_gadgets=[ap2t, cl2t],
        target_gadget=target,
        strategy=SearchStrategy.BFS,
        max_depth=4,
        verbose=True
    )
    
    if path:
        print("Found solution!")
        print("Operations:")
        for op in path:
            print(f"  {format_operation(op)}")
        return True
    else:
        print("No solution found")
        return False

def run_all_tests():
    """Run all simulation search tests"""
    tests = [
        test_ap2t_sim_c2t,
        test_cl2t_sim_pl2t,
        test_c2t_sim_p2t,
        test_door_sim_scd,
        test_apl2t_sim_cl2t,
        test_complex_simulation
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
        print(f"Test {test.__name__}: {'PASSED' if result else 'FAILED'}")
    
    print("\nSummary:")
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {sum(results)}")
    print(f"Failed: {len(tests) - sum(results)}")
    
    return all(results)

if __name__ == "__main__":
    run_all_tests() 