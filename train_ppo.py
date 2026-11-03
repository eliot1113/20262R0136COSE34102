"""
Student-side short PPO training using the official FinRL GitHub environment.

This script uses:
  official_repo_here/FinRL_Contest_2025/Task_1_FinRL_DeepSeek_Stock/env_stocktrading.py

It does NOT define or use a custom environment.

Run from project root:

  python -m student_submission.train_ppo --start_date 2013-01-01 --end_date 2018-12-31 --total_timesteps 2000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO

from test import load_market_dataframe, make_official_finrl_env


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--start_date", type=str, default="2013-01-01")
    parser.add_argument("--end_date", type=str, default="2018-12-31")
    parser.add_argument("--total_timesteps", type=int, default=2000)

    parser.add_argument(
        "--finrl_repo_path",
        type=str,
        default=None,
        help="Path to official FinRL_Contest_2025 repo.",
    )

    parser.add_argument(
        "--data_file",
        type=str,
        default="train_data_2013_2018.csv",
        help="HuggingFace file inside benstaf/nasdaq_2013_2023.",
    )

    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--local_csv", type=str, default=None)

    args = parser.parse_args()

    df = load_market_dataframe(
        split=args.split,
        data_file=args.data_file,
        local_csv=args.local_csv,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    env = make_official_finrl_env(
        df=df,
        finrl_repo_path=args.finrl_repo_path,
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        n_steps=128,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        device="cpu",
    )

    model.learn(total_timesteps=args.total_timesteps)

    save_path = Path("student_submission") / "saved_agent"
    model.save(save_path)

    print(f"Saved PPO agent to: {save_path}.zip")


if __name__ == "__main__":
    main()
