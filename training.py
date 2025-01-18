import os
from datetime import datetime
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
from sumo_rl import SumoEnvironment

# Paths to the required files
net_file = 'C:/Programming/BYOP/PALO ALTO/osm.net.xml'
route_file = 'C:/Programming/BYOP/PALO ALTO/osm.rou.xml'

# Generate unique directories for this training session
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_csv_dir = f'C:/Programming/BYOP/Training/CSV_Outputs/session_{timestamp}'
tensorboard_log_dir = f'C:/Programming/BYOP/Training/TensorBoard/session_{timestamp}'
saved_models_dir = f'C:/Programming/BYOP/Training/Saved_Models/session_{timestamp}'

os.makedirs(output_csv_dir, exist_ok=True)
os.makedirs(tensorboard_log_dir, exist_ok=True)
os.makedirs(saved_models_dir, exist_ok=True)

# Traffic management thresholds and rewards
CONGESTION_THRESHOLD = 100
WAITING_TIME_THRESHOLD = 100
CONGESTION_PENALTY = -100
REWARD_FOR_SURVIVAL = 1
SIGNAL_COOLDOWN_STEPS = 5

class CongestionWrapper(SumoEnvironment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.steps_since_signal_change = 0
        self.last_action = None

    def step(self, action):
        # Cooldown logic
        if self.steps_since_signal_change < SIGNAL_COOLDOWN_STEPS:
            action = self.last_action if self.last_action is not None else action

        obs, reward, terminated, truncated, info = super().step(action)

        self.steps_since_signal_change += 1
        if action != self.last_action:
            self.steps_since_signal_change = 0

        # Apply survival reward and congestion penalty
        total_vehicles = info.get('system_total_stopped', 0)
        avg_wait_time = info.get('system_mean_waiting_time', 0)
        reward += REWARD_FOR_SURVIVAL
        if total_vehicles > CONGESTION_THRESHOLD or avg_wait_time > WAITING_TIME_THRESHOLD:
            reward += CONGESTION_PENALTY
            terminated = True

        # Log metrics for TensorBoard
        info.update({
            'total_vehicles': total_vehicles,
            'avg_wait_time': avg_wait_time,
            'step_reward': reward,
            'reward/survival_reward': REWARD_FOR_SURVIVAL,
            'reward/congestion_penalty': CONGESTION_PENALTY if total_vehicles > CONGESTION_THRESHOLD else 0,
        })

        self.last_action = action
        return obs, reward, terminated, truncated, info


if __name__ == "__main__":
    try:
        # Initialize the environment
        env = CongestionWrapper(
            net_file=net_file,
            route_file=route_file,
            out_csv_name=os.path.join(output_csv_dir, 'output.csv'),
            single_agent=True,
            use_gui=False,  # Set to True for GUI visualization
            num_seconds=100000,
        )

        # Wrap the environment for RL
        vec_env = DummyVecEnv([lambda: env])

        # Configure logger for TensorBoard
        logger = configure(tensorboard_log_dir, ["stdout", "tensorboard"])
        env.logger = logger

        # Callbacks for model saving
        eval_callback = EvalCallback(
            vec_env,
            best_model_save_path=os.path.join(saved_models_dir, "best_models"),
            log_path=saved_models_dir,
            eval_freq=5000,
            deterministic=True,
            render=False
        )
        checkpoint_callback = CheckpointCallback(
            save_freq=5000,
            save_path=saved_models_dir,
            name_prefix="checkpoint"
        )

        # Define the DQN model
        model = DQN(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=0.0005,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=64,
            tau=1.0,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1000,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            exploration_fraction=0.2,
            verbose=1,
            tensorboard_log=tensorboard_log_dir,
        )
        model.set_logger(logger)

        # Train the model
        model.learn(
            total_timesteps=50000,
            tb_log_name="DQN_Traffic_Control",
            callback=[eval_callback, checkpoint_callback],
        )

        # Save the final model
        model.save(os.path.join(saved_models_dir, 'final_model.zip'))
        print(f"Training completed. Model and logs saved to {saved_models_dir}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        env.close()
        print("Environment closed.")
