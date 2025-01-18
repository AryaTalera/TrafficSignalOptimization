import os
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from sumo_rl import SumoEnvironment

# Validate file paths
files = [
    'C:/Programming/BYOP/Dehli - Barlow Moor Road - Easy/osm.net.xml.gz',
    'C:/Programming/BYOP/Dehli - Barlow Moor Road - Easy/osm.rou.xml'
]
for file in files:
    if not os.path.exists(file):
        raise FileNotFoundError(f"File not found: {file}")

# Congestion thresholds
CONGESTION_THRESHOLD = 30  # Max total stopped vehicles allowed
WAITING_TIME_THRESHOLD = 50  # Max avg waiting time (seconds)
CONGESTION_PENALTY = -10    # Penalty for exceeding thresholds

class CongestionWrapper(SumoEnvironment):
    def step(self, action):
        # Take a step in the environment
        obs, reward, terminated, truncated, info = super().step(action)

        # Monitor congestion metrics
        total_vehicles = info.get('system_total_stopped', 0)
        average_waiting_time = info.get('system_mean_waiting_time', 0)

        # Apply penalty for excessive congestion
        if total_vehicles > CONGESTION_THRESHOLD or average_waiting_time > WAITING_TIME_THRESHOLD:
            reward += CONGESTION_PENALTY
            terminated = True  # End the episode early

        return obs, reward, terminated, truncated, info

if __name__ == "__main__":
    try:
        # Create SUMO environment with congestion handling
        env = CongestionWrapper(
            net_file=files[0],
            route_file=files[1],
            out_csv_name="C:/Programming/BYOP/Dehli - Barlow Moor Road - Easy/output.csv",
            single_agent=True,
            use_gui=False,
            num_seconds=100000,
        )

        vec_env = DummyVecEnv([lambda: env])

        # Define the DQN model
        model = DQN(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=0.001,
            learning_starts=100,
            train_freq=1,
            target_update_interval=500,
            exploration_initial_eps=0.1,
            exploration_final_eps=0.01,
            verbose=1,
        )

        # Train the model
        model.learn(total_timesteps=100000)
        model.save('C:/Programming/BYOP/Training/Saved_Models/model-1')
        print("Training completed and model saved.")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        env.close()
        print("Environment closed.")
