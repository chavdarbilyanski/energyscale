import mlflow
import pickle
import os
import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

class CorrectShapeEnv(gym.Env):
    def __init__(self):
        super(CorrectShapeEnv, self).__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    def step(self, action):
        return self.observation_space.sample(), 0, False, False, {}

    def reset(self, seed=None, options=None):
        return self.observation_space.sample(), {}

class StableBaselinesAgentWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        print("Loading model and environment normalization stats...")
        agent_path = context.artifacts["agent_zip"]
        stats_path = context.artifacts["vecnormalize_stats"]
        dummy_env = DummyVecEnv([lambda: CorrectShapeEnv()])
        self.vec_env = VecNormalize.load(stats_path, dummy_env)
        self.vec_env.training = False
        self.vec_env.norm_reward = False
        self.agent = PPO.load(agent_path, env=self.vec_env)
        print("Model and stats loaded successfully.")

    def predict(self, context, model_input):
        observation = model_input.values
        action, _states = self.agent.predict(observation, deterministic=True)
        return pd.Series(action)

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    experiment_name = "Battery RL Agents"

    # Ensure experiment exists
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        experiment_id = mlflow.create_experiment(experiment_name, artifact_location="file:///Users/chavdarbilyanski/energyscale/mlruns/artifacts")
        print(f"Created new experiment '{experiment_name}' with ID: {experiment_id}")
    else:
        experiment_id = experiment.experiment_id
        print(f"Using existing experiment '{experiment_name}' with ID: {experiment_id}")

    # Use absolute paths directly
    AGENT_FILE_PATH = "/Users/chavdarbilyanski/energyscale/battery_ppo_agent_v2.zip"
    STATS_FILE_PATH = "/Users/chavdarbilyanski/energyscale/vec_normalize_stats_v2.pkl"

    # Validate file existence
    if not os.path.exists(AGENT_FILE_PATH):
        raise FileNotFoundError(f"Agent file not found at {AGENT_FILE_PATH}")
    if not os.path.exists(STATS_FILE_PATH):
        raise FileNotFoundError(f"Stats file not found at {STATS_FILE_PATH}")

    with mlflow.start_run(run_name="Upload PPO Agent v2 with VecNormalize", experiment_id=experiment_id) as run:
        print(f"MLflow Run ID: {run.info.run_id}")
        
        artifacts = {
            "agent_zip": AGENT_FILE_PATH,
            "vecnormalize_stats": STATS_FILE_PATH
        }
        
        pip_requirements = [
            f"mlflow==2.15.1",  # Match your current MLflow version
            "stable-baselines3>=2.0.0",
            "gymnasium",
            "torch",
            "pandas",
            "numpy"
        ]

        # Log the pyfunc model
        try:
            mlflow.pyfunc.log_model(
                artifact_path="ppo_battery_model2",
                python_model=StableBaselinesAgentWrapper(),
                artifacts=artifacts,
                pip_requirements=pip_requirements
            )
            print("Model logged successfully.")
        except Exception as e:
            print(f"Failed to log model: {e}")
            raise
        
        # Register the model
        model_uri = f"runs:/{run.info.run_id}/ppo_battery_model2"
        registered_model = mlflow.register_model(model_uri, "BatteryPPOModel")

        # Set tags and stage
        client = mlflow.MlflowClient()
        client.set_model_version_tag(
            name="BatteryPPOModel",
            version=registered_model.version,
            key="bess_arbitrage",
            value="production-ready"
        )
        client.transition_model_version_stage(
            name="BatteryPPOModel",
            version=registered_model.version,
            stage="Production"
        )

        print(f"Model registered as BatteryPPOModel version {registered_model.version} with tag 'bess_arbitrage: production-ready' and staged to Production.")