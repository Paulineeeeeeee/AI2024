"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).

Please only change the parts of the file you are asked to.  Look for the lines
that say

"*** YOUR CODE HERE ***"

Follow the project description for details.

Good luck and happy searching!
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    print("Solution:", [s, s, w, s, w, w, s, w])
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    """
    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    
    "*** YOUR CODE HERE ***"
    # Initialize the stack with the starting point. The stack will store tuples of (state, path).
    stack = util.Stack()
    start_state = problem.getStartState()
    stack.push((start_state, []))

    # Initialize a set to keep track of visited nodes
    visited = set()

    while not stack.isEmpty():
        # Pop a state and path from the stack
        current_state, path = stack.pop()

        # If the current state is the goal, return the path
        if problem.isGoalState(current_state):
            return path

        # If the current state has not been visited, explore it
        if current_state not in visited:
            visited.add(current_state)

            # For each successor, push the new state and path to the stack
            for successor, action, _ in problem.getSuccessors(current_state):
                if successor not in visited:
                    new_path = path + [action]
                    stack.push((successor, new_path))

    # If the loop ends without returning, no path was found
    return []
    util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # 使用 util 中的 Queue 來存儲待探索的節點及其路徑
    queue = util.Queue()
    start_state = problem.getStartState()
    queue.push((start_state, []))

    # 使用集合來記錄已經訪問過的節點
    visited = set()

    while not queue.isEmpty():
        # 從隊列中取出一個節點和到達該節點的路徑
        current_state, path = queue.pop()

        # 如果這個節點是目標狀態，則返回路徑
        if problem.isGoalState(current_state):
            return path

        # 如果這個節點還沒有被訪問過，則探索它
        if current_state not in visited:
            visited.add(current_state)

            # 對於當前節點的每一個後繼節點，如果它還沒有被訪問過，則將其添加到隊列中
            for successor, action, _ in problem.getSuccessors(current_state):
                if successor not in visited:
                    new_path = path + [action]  # 生成到達後繼節點的新路徑
                    queue.push((successor, new_path))

    # 如果搜索完成而沒有找到解，則返回空列表
    return []
    util.raiseNotDefined()

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    """Search the node of least total cost first."""
    # PriorityQueue 用於存儲待探索的節點以及到達這些節點的總成本
    # 每個節點是一個元組，包含當前狀態和到達該狀態的路徑（動作序列）
    priorityQueue = util.PriorityQueue()
    start_state = problem.getStartState()
    priorityQueue.push((start_state, [], 0), 0) # 初始狀態，路徑為空，成本為 0

    # 使用集合來記錄已經訪問過的節點，避免重複探索
    visited = set()

    while not priorityQueue.isEmpty():
        # 從優先隊列中取出成本最低的節點
        current_state, path, current_cost = priorityQueue.pop()

        # 檢查當前節點是否是目標狀態
        if problem.isGoalState(current_state):
            return path

        # 如果當前節點未被探索過，則檢查其所有後繼節點
        if current_state not in visited:
            visited.add(current_state)

            for successor, action, cost in problem.getSuccessors(current_state):
                if successor not in visited:
                    # 計算到達後繼節點的總成本
                    new_cost = current_cost + cost
                    new_path = path + [action]
                    priorityQueue.push((successor, new_path, new_cost), new_cost)

    # 如果搜索完成而沒有找到解，則返回空列表
    return []
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    """Search the node that has the lowest combined cost and heuristic first."""
    # PriorityQueue 用於存儲待探索的節點以及到達這些節點的評估函數值
    priorityQueue = util.PriorityQueue()
    start_state = problem.getStartState()
    # 初始狀態，路徑為空，成本為 0。評估函數值是成本加上啟發式值
    priorityQueue.push((start_state, [], 0), heuristic(start_state, problem))

    # 使用集合記錄已訪問的節點，避免重複探索
    visited = set()

    while not priorityQueue.isEmpty():
        # 從優先隊列中取出評估函數值最低的節點
        current_state, path, current_cost = priorityQueue.pop()

        # 檢查當前節點是否是目標狀態
        if problem.isGoalState(current_state):
            return path

        # 如果當前節點未被探索過，則檢查其所有後繼節點
        if current_state not in visited:
            visited.add(current_state)

            for successor, action, cost in problem.getSuccessors(current_state):
                if successor not in visited:
                    # 計算到達後繼節點的總成本和評估函數值
                    new_cost = current_cost + cost
                    new_path = path + [action]
                    # f(n) = g(n) + h(n)
                    f_value = new_cost + heuristic(successor, problem)
                    priorityQueue.push((successor, new_path, new_cost), f_value)

    # 如果搜索完成而沒有找到解，則返回空列表
    return []
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
