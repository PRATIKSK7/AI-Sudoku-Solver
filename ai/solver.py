def solve(bo, solutions_found=None, limit=2, iterations=None, max_iterations=50000):
    """
    Solves a sudoku board using backtracking and finds up to `limit` solutions.
    Returns the number of solutions found.
    Raises TimeoutError if max_iterations is exceeded.
    """
    if solutions_found is None:
        solutions_found = [0]
    if iterations is None:
        iterations = [0]
        
    iterations[0] += 1
    if iterations[0] > max_iterations:
        raise TimeoutError(f"Solver exceeded maximum iterations ({max_iterations})")
        
    find = find_empty(bo)
    if not find:
        solutions_found[0] += 1
        return solutions_found[0]
        
    row, col = find

    for i in range(1, 10):
        if valid(bo, i, (row, col)):
            bo[row][col] = i
            
            solve(bo, solutions_found, limit, iterations, max_iterations)
            if solutions_found[0] >= limit:
                return solutions_found[0]
                
            bo[row][col] = 0

    return solutions_found[0]

def valid(bo, num, pos):
    """
    Returns if the attempted move is valid
    """
    # Check row
    for i in range(len(bo[0])):
        if bo[pos[0]][i] == num and pos[1] != i:
            return False

    # Check column
    for i in range(len(bo)):
        if bo[i][pos[1]] == num and pos[0] != i:
            return False

    # Check box
    box_x = pos[1] // 3
    box_y = pos[0] // 3

    for i in range(box_y * 3, box_y * 3 + 3):
        for j in range(box_x * 3, box_x * 3 + 3):
            if bo[i][j] == num and (i, j) != pos:
                return False

    return True

def find_empty(bo):
    """
    finds an empty space in the board
    """
    for i in range(len(bo)):
        for j in range(len(bo[0])):
            if bo[i][j] == 0:
                return (i, j)
    return None

def get_board_solution(board):
    """
    Wrapper for solve function.
    Returns: Tuple(status_string, solved_board_or_None)
    status_string can be: "SOLVED", "UNSOLVABLE", "MULTIPLE_SOLUTIONS"
    """
    if len(board) == 81:
        bo = [board[i*9 : (i+1)*9] for i in range(9)]
    else:
        # Create deep copy
        bo = [row[:] for row in board]

    # First pass: count solutions (mutates bo)
    check_bo = [row[:] for row in bo]
    try:
        solutions_count = solve(check_bo, limit=2)
    except TimeoutError:
        print("Solver Timeout: Maximum iterations reached. Treating as UNSOLVABLE.")
        return "UNSOLVABLE", None
    
    if solutions_count == 0:
        return "UNSOLVABLE", None
    elif solutions_count > 1:
        return "MULTIPLE_SOLUTIONS", None
    else:
        # Second pass: solve cleanly on a fresh copy
        solve_bo = [row[:] for row in bo]
        try:
            solve(solve_bo, limit=1)
        except TimeoutError:
            return "UNSOLVABLE", None
            
        if len(board) == 81:
            flat = [item for row in solve_bo for item in row]
            return "SOLVED", flat
        return "SOLVED", solve_bo

