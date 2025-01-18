import gymnasium as gym
import time
from stable_baselines3 import DQN
import matplotlib.pyplot as plt

# Validate file paths
net_file = 'C:/Programming/BYOP/PALO ALTO/osm.net.xml'
route_file = 'C:/Programming/BYOP/PALO ALTO/osm.rou.xml'

# Congestion thresholds and rewards
CONGESTION_THRESHOLD = 100  # Maximum total stopped vehicles allowed
CONGESTION_PENALTY = -100   # Penalty for exceeding congestion
WAITING_TIME_THRESHOLD = 100  # Maximum average waiting time (seconds) allowed
REWARD_FOR_SURVIVAL = 1     # Reward for surviving each step
SIGNAL_COOLDOWN_STEPS = 5  # Cooldown for signal change

# Create environment
env = gym.make(
    'sumo-rl-v0',
    net_file=net_file,
    route_file=route_file,
    use_gui=True,
    num_seconds=5000
)

# Load the trained model
model = DQN.load('C:/Programming/BYOP/Training/Saved_Models/session_20250118_210749/checkpoint_5000_steps.zip', env=env)

# Initialize variables
last_action = env.action_space.sample()
steps_since_signal_change = 0
done = False
step_count = 0
total_reward = 0
total_vehicles = []
avg_waiting_times = []

# Evaluate the model
try:
    obs, info = env.reset()
    print("Initial Observation:", obs)
    print("Initial Info:", info)

    while not done:
        time.sleep(0.5)

        # Allow action if cooldown has passed
        if steps_since_signal_change >= SIGNAL_COOLDOWN_STEPS:
            action, _ = model.predict(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            if action != last_action:
                steps_since_signal_change = 0  # Reset cooldown
            last_action = action
        else:
            action = last_action
            next_obs, reward, terminated, truncated, info = env.step(action)

        steps_since_signal_change += 1

        # Apply congestion penalty or survival reward
        total_vehicles.append(info.get('system_total_stopped', 0))
        avg_waiting_times.append(info.get('system_mean_waiting_time', 0))
        reward += REWARD_FOR_SURVIVAL
        if total_vehicles[-1] > CONGESTION_THRESHOLD or avg_waiting_times[-1] > WAITING_TIME_THRESHOLD:
            print(f"Congestion detected! Vehicles: {total_vehicles[-1]}, Avg wait time: {avg_waiting_times[-1]}. Penalizing.")
            reward += CONGESTION_PENALTY
            done = True

        total_reward += reward
        print(f"Step: {step_count}, Action: {action}, Reward: {reward}, Vehicles: {total_vehicles[-1]}, Avg Wait: {avg_waiting_times[-1]}")
        step_count += 1
        done = done or terminated or truncated

finally:
    env.close()
    print("Environment closed successfully.")

print(f"Total steps: {step_count}, Total reward: {total_reward}")

# Plot the results
plt.figure(figsize=(12, 6))

# Total stopped vehicles
plt.subplot(1, 2, 1)
plt.plot(total_vehicles, label='Total Stopped Vehicles')
plt.xlabel('Step')
plt.ylabel('Total Stopped Vehicles')
plt.title('Total Stopped Vehicles Over Time')
plt.legend()

# Average waiting time
plt.subplot(1, 2, 2)
plt.plot(avg_waiting_times, label='Average Waiting Time', color='orange')
plt.xlabel('Step')
plt.ylabel('Average Waiting Time (s)')
plt.title('Average Waiting Time Over Time')
plt.legend()

plt.tight_layout()
plt.show()