from flask import Flask, render_template, request, jsonify
import numpy as np
import random

app = Flask(__name__)

class GridWorld:
    def __init__(self, n):
        self.n = n
        self.grid = np.zeros((n, n))
        self.start = None
        self.end = None
        self.obstacles = []
        self.policy = {}
        self.value_function = {}

    def set_start(self, row, col):
        self.start = (row, col)

    def set_end(self, row, col):
        self.end = (row, col)

    def set_obstacle(self, row, col):
        self.obstacles.append((row, col))

    def initialize_random_policy(self):
        """Initialize a random policy for all states."""
        for i in range(self.n):
            for j in range(self.n):
                state = (i, j)
                if state not in self.obstacles and state != self.end:
                    self.policy[str(state)] = random.choice(['up', 'down', 'left', 'right'])


    def get_possible_actions(self, state):
        """Get the possible actions from a given state."""
        actions = []
        row, col = state
        if row > 0 and (row - 1, col) not in self.obstacles:
            actions.append('up')
        if row < self.n - 1 and (row + 1, col) not in self.obstacles:
            actions.append('down')
        if col > 0 and (row, col - 1) not in self.obstacles:
            actions.append('left')
        if col < self.n - 1 and (row, col + 1) not in self.obstacles:
            actions.append('right')
        return actions

    def bellman_update(self, state, action, gamma=0.9):
        row, col = map(int, state[1:-1].split(', '))  # 将字符串转换为行和列
        value = 0
        next_states = []
        transition_probs = []
        if action == 'up':
            next_states.append((row - 1, col))
            transition_probs.append(1.0)
        elif action == 'down':
            next_states.append((row + 1, col))
            transition_probs.append(1.0)
        elif action == 'left':
            next_states.append((row, col - 1))
            transition_probs.append(1.0)
        elif action == 'right':
            next_states.append((row, col + 1))
            transition_probs.append(1.0)

        for i, next_state in enumerate(next_states):
            next_state_str = str(next_state)  # 将元组转换为字符串
            if next_state_str == self.end:
                value += transition_probs[i] * 1  # terminal reward
            elif next_state_str in map(str, self.obstacles):
                value += transition_probs[i] * -0.1  # obstacle penalty
            else:
                value += transition_probs[i] * (gamma * self.value_function.get(next_state_str, 0) - 0.1)  # step penalty

        self.value_function[state] = value

    def value_iteration(self, gamma=0.9, epsilon=1e-6):
        """Perform value iteration to converge the value function."""
        self.initialize_random_policy()
        delta = float('inf')
        while delta > epsilon:
            delta = 0
            for state in self.policy:
                if state != self.end:
                    action = self.policy[state]
                    old_value = self.value_function.get(state, 0)
                    self.bellman_update(state, action, gamma)
                    new_value = self.value_function[state]
                    delta = max(delta, abs(old_value - new_value))
                    print(f"State {state}: Value = {new_value}")  # 打印当前状态值

    def get_optimal_policy(self):
        """Derive the optimal policy from the converged value function."""
        optimal_policy = {}
        for state_str in self.policy:
            row, col = map(int, state_str[1:-1].split(', '))
            state = (row, col)
            if state != self.end:
                max_value = -float('inf')
                best_action = None
                for action in self.get_possible_actions(state):
                    next_states = []
                    transition_probs = []
                    if action == 'up':
                        next_states.append((row - 1, col))
                        transition_probs.append(1.0)
                    elif action == 'down':
                        next_states.append((row + 1, col))
                        transition_probs.append(1.0)
                    elif action == 'left':
                        next_states.append((row, col - 1))
                        transition_probs.append(1.0)
                    elif action == 'right':
                        next_states.append((row, col + 1))
                        transition_probs.append(1.0)

                    value = sum(transition_probs[i] * self.value_function.get(str(next_state), 0) for i, next_state in enumerate(next_states))
                    if value > max_value:
                        max_value = value
                        best_action = action
                optimal_policy[state_str] = best_action
        return optimal_policy


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_grid', methods=['POST'])
def generate_grid():
    n = int(request.form['n'])
    if n < 3 or n > 7:
        return "Error: n should be between 3 and 7"
    return render_template('index.html', n=n)

@app.route('/evaluate_policy', methods=['POST'])
def evaluate_policy():
    points = request.json['points']
    n = int(request.json['n'])

    grid_world = GridWorld(n)

    for point in points:
        row, col, cell_type = point['row'], point['col'], point['type']
        if cell_type == 'start':
            grid_world.set_start(row, col)
        elif cell_type == 'end':
            grid_world.set_end(row, col)
        elif cell_type == 'obstacle':
            grid_world.set_obstacle(row, col)

    grid_world.value_iteration()
    optimal_policy = grid_world.get_optimal_policy()

    return jsonify({
        'value_function': grid_world.value_function,
        'optimal_policy': optimal_policy
    })



if __name__ == '__main__':
    app.run(debug=True)