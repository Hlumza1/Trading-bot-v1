import numpy as np
import random

# Initialize the environment parameters (states, actions)
# Let's assume we have 5 states (can represent market conditions, price levels, etc.)
# And 3 actions (buy, sell, hold)
states = 5
actions = 3

# Q-table: Rows = States, Columns = Actions (initialized with zeros)
q_table = np.zeros((states, actions))

# Learning parameters
learning_rate = 0.1   # How much new info overrides old info
discount_factor = 0.95  # How much future rewards are valued compared to immediate rewards
epsilon = 1.0  # Epsilon-greedy factor for exploration vs. exploitation
epsilon_decay = 0.995  # Gradual decrease in exploration
min_epsilon = 0.01  # Minimum exploration rate

# Reward function: Example rewards for actions in a certain state
# Modify this based on your needs (e.g., reward buying in low prices, selling high, etc.)
reward_matrix = [
    [0, -10, 10],  # State 0: Rewards for (buy, sell, hold)
    [-10, 0, 5],   # State 1: Slightly different rewards based on market condition
    [5, 10, 0],    # State 2: Higher reward for buying, holding
    [-5, 20, 0],   # State 3: Reward selling during price peaks
    [0, 5, -5]     # State 4: Negative reward for wrong decisions
]

# Training the agent with Q-learning
episodes = 1000  # Number of iterations
for episode in range(episodes):
    state = random.randint(0, states-1)  # Start with a random state

    while True:
        # Exploration vs. Exploitation decision
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, actions-1)  # Explore: Random action
        else:
            action = np.argmax(q_table[state])  # Exploit: Choose action with highest Q-value

        # Simulate environment feedback (reward) for the action taken
        reward = reward_matrix[state][action]

        # Simulate the new state after taking action (for simplicity, assume new random state)
        new_state = random.randint(0, states-1)

        # Update Q-value: Q(state, action) = old value + learning_rate * (reward + discount * max future Q - old value)
        q_table[state, action] = q_table[state, action] + learning_rate * (reward + discount_factor * np.max(q_table[new_state]) - q_table[state, action])

        # Break if the new state is terminal (for simplicity, assume end of episode)
        if new_state == states - 1:
            break

        # Update state to the new state
        state = new_state

    # Decay epsilon (less exploration over time)
    if epsilon > min_epsilon:
        epsilon *= epsilon_decay

# Testing the learned policy
print("Trained Q-table:")
print(q_table)

# Now the bot can make decisions based on the trained Q-table.
# For example:
current_state = 0  # Assume starting in state 0
best_action = np.argmax(q_table[current_state])
actions_names = ['Buy', 'Sell', 'Hold']
print(f"Best action for state {current_state}: {actions_names[best_action]}")
