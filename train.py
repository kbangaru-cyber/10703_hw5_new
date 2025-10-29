import os, sys, types
import numpy as np
import torch
import matplotlib.pyplot as plt

# ---------- Gym / Gymnasium compatibility ----------

import gym  # <= you are here (Gym)
_USING_GYMNASIUM = False
# Create a tiny "gymnasium" shim so envs/__init__.py can import register from it
try:
    from gym.envs.registration import register as _gym_register
    gymnasium = types.ModuleType("gymnasium")
    gymnasium.envs = types.ModuleType("gymnasium.envs")
    gymnasium.envs.registration = types.ModuleType("gymnasium.envs.registration")
    gymnasium.envs.registration.register = _gym_register
    sys.modules["gymnasium"] = gymnasium
    sys.modules["gymnasium.envs"] = gymnasium.envs
    sys.modules["gymnasium.envs.registration"] = gymnasium.envs.registration
except Exception:
    pass

# Helpers so code runs on both gym and gymnasium APIs
def _reset(env):
    out = env.reset()
    return out[0] if isinstance(out, tuple) else out

def _step(env, action):
    out = env.step(action)
    if isinstance(out, tuple) and len(out) == 5:  # gymnasium
        obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
        return obs, reward, done, info
    return out  # gym: (obs, reward, done, info)

# ---------- Local imports (envs registers Pushing2D) ----------
import envs  # now succeeds because we shimmied gymnasium.register above
import td3
import replay_buffer
from model import PENN

# Pull constants if available; otherwise defaults
try:
    from run import LR, STATE_DIM
except Exception:
    LR = 1e-3
    STATE_DIM = 8  # first 8 dims = dynamics; last 2 = goal


class TrainerTD3:
    def __init__(
        self,
        env_name,
        seed,
        start_timesteps,
        eval_freq,
        max_timesteps,
        expl_noise,
        batch_size,
        num_nets,
        device,
    ):
        self.env = gym.make(env_name)
        self.device = device

        state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.max_action = float(self.env.action_space.high[0])
        self.expl_noise = expl_noise

        # Load learned dynamics ensemble (like PETS)
        self.model = PENN(num_nets, STATE_DIM, self.action_dim, LR, device=device)
        self.model.load_state_dict(torch.load("model.pt", map_location=device))
        self.model.eval()

        # Seeding
        try:
            self.env.seed(seed)
            self.env.action_space.seed(seed)
        except Exception:
            pass
        torch.manual_seed(seed)
        np.random.seed(seed)

        # TD3 + buffers
        self.policy = td3.TD3(state_dim, self.action_dim, self.max_action, device)
        self.memory = replay_buffer.ReplayBuffer(state_dim, self.action_dim)
        self.real_memory = replay_buffer.ReplayBuffer(
            state_dim, self.action_dim, max_size=5000, save_timestep=True
        )

        self.start_timesteps = start_timesteps
        self.eval_freq = eval_freq
        self.batch_size = batch_size
        self.max_timesteps = max_timesteps

    # -------------------- Eval --------------------
    def eval_policy(self, eval_episodes=10):
        avg_reward = 0.0
        successes = 0
        for _ in range(eval_episodes):
            state, done = _reset(self.env), False
            while not done:
                action = self.policy.select_action(np.array(state))
                state, reward, done, info = _step(self.env, action)
                avg_reward += reward
                if isinstance(info, dict) and info.get("done") == "goal reached":
                    successes += 1
        avg_reward /= eval_episodes
        success_rate = successes / eval_episodes
        print("---------------------------------------")
        print(
            f"Evaluation over {eval_episodes} episodes: "
            f"Reward {avg_reward:.3f}, Success Rate {success_rate:.2f}"
        )
        print("---------------------------------------")
        return avg_reward, success_rate

    # -------------------- Synthetic transition (TS1) --------------------
    def get_synthetic_transition(self, state, action):
        """
        One-step hallucination using the learned ensemble (TS1):
          - Predict Δs on first STATE_DIM dims.
          - Keep goal dims (last 2) unchanged.
          - Return env's transition by stepping to the synthetic next full state.
        """
        s8 = np.asarray(state[:STATE_DIM], dtype=np.float32)[None, :]
        a = np.asarray(action, dtype=np.float32)[None, :]
        sa = np.concatenate([s8, a], axis=1)
        sa_t = torch.tensor(sa, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            outs = self.model(sa_t)  # list[(mean, logvar)] length = num_nets
            n = np.random.randint(self.model.num_nets) if self.model.num_nets > 1 else 0
            mean, logvar = outs[n]
            std = torch.exp(0.5 * logvar)
            delta = mean + torch.randn_like(mean) * std

        dyn_next = (s8 + delta.detach().cpu().numpy()).squeeze(0)  # [STATE_DIM]
        full_next = np.concatenate(
            [dyn_next, np.asarray(state[STATE_DIM:], dtype=np.float32)], axis=0
        ).astype(np.float32)

        next_state, reward, done, info = self.env.step_state(full_next.tolist())
        return next_state, reward, done, info

    # -------------------- 1.4.1: Mix synthetic vs real --------------------
    def train_td3_with_mix(self, synthetic_ratio):
        state, done = _reset(self.env), False
        episode_reward = 0.0
        episode_num = 0
        ep_steps = 0

        eval_rewards, eval_success_rates = [], []

        for t in range(1, int(self.max_timesteps) + 1):
            ep_steps += 1

            # Action
            if t < self.start_timesteps:
                action = self.env.action_space.sample()
            else:
                action = (
                    self.policy.select_action(np.array(state))
                    + np.random.normal(
                        0, self.max_action * self.expl_noise, size=self.action_dim
                    )
                ).clip(-self.max_action, self.max_action)

            # Real or synthetic transition
            if np.random.rand() < synthetic_ratio:
                next_state, reward, done, _ = self.get_synthetic_transition(state, action)
            else:
                next_state, reward, done, _ = _step(self.env, action)

            # Store and train
            self.memory.add(state, action, next_state, reward, done)
            state = next_state
            episode_reward += reward

            if t >= self.start_timesteps:
                self.policy.train(self.memory, self.batch_size)

            if done:
                print(
                    f"Total T: {t+1} Ep: {episode_num+1} Ep_T: {ep_steps} Reward: {episode_reward:.3f}"
                )
                state, done = _reset(self.env), False
                episode_reward, ep_steps = 0.0, 0
                episode_num += 1

            if t % self.eval_freq == 0:
                avg_rew, success_rate = self.eval_policy()
                eval_rewards.append(avg_rew)
                eval_success_rates.append(success_rate)

        print("Training finished.")
        return eval_rewards, eval_success_rates

    # -------------------- 1.4.2: Rollouts from real mid-states --------------------
    def train_with_model_rollouts(self, rollout_length):
        eval_rewards, eval_success_rates = [], []

        # Warm up a pool of real transitions (save timestep)
        state, done = _reset(self.env), False
        for _ in range(self.start_timesteps * 2):
            action = self.env.action_space.sample()
            next_state, reward, done, _ = _step(self.env, action)
            # elapsed_steps is maintained by env; subtract 1 to get step index for s->s'
            ts = getattr(self.env, "elapsed_steps", 0)
            self.real_memory.add(state, action, next_state, reward, done, timestep=ts - 1)
            state = next_state
            if done:
                state, done = _reset(self.env), False

        t = self.start_timesteps
        while t < self.max_timesteps:
            # Collect one real episode (also saved to real_memory)
            state, done = _reset(self.env), False
            while not done:
                action = (
                    self.policy.select_action(np.array(state))
                    + np.random.normal(
                        0, self.max_action * self.expl_noise, size=self.action_dim
                    )
                ).clip(-self.max_action, self.max_action)
                next_state, reward, done, _ = _step(self.env, action)
                ts = getattr(self.env, "elapsed_steps", 0)
                self.real_memory.add(state, action, next_state, reward, done, timestep=ts - 1)
                state = next_state

            # Sample a real mid-state (with its timestep)
            sample = self.real_memory.sample(1)
            # sample is [state, action, next_state, reward, not_done, timestep] when save_timestep=True
            model_state = sample[0].squeeze(0).cpu().numpy()
            timestep = sample[5]
            # robust cast to an int
            timestep = int(np.array(timestep).squeeze()) if not np.isscalar(timestep) else int(timestep)

            # Set env to that state
            try:
                self.env.set_state(model_state.tolist(), elapsed_steps=timestep)
            except TypeError:
                self.env.set_state(model_state.tolist())

            # Synthetic rollout of fixed length
            for _ in range(rollout_length):
                model_action = (
                    self.policy.select_action(np.array(model_state))
                    + np.random.normal(
                        0, self.max_action * self.expl_noise, size=self.action_dim
                    )
                ).clip(-self.max_action, self.max_action)

                next_model_state, model_reward, model_done, _ = self.get_synthetic_transition(
                    model_state, model_action
                )

                self.memory.add(model_state, model_action, next_model_state, model_reward, model_done)
                model_state = next_model_state
                t += 1

                self.policy.train(self.memory, self.batch_size)

                if t % self.eval_freq == 0:
                    print(f"T: {t}")
                    avg_rew, success_rate = self.eval_policy()
                    eval_rewards.append(avg_rew)
                    eval_success_rates.append(success_rate)

                if model_done:
                    break

        return eval_rewards, eval_success_rates

    # -------------------- 1.4.3: Model error vs rollout length --------------------
    def compute_and_plot_model_error(self, max_rollout_length=15, num_trajectories=50):
        all_errors = []

        for _ in range(num_trajectories):
            real_state, done = _reset(self.env), False
            real_trajectory = []

            # Build a real trajectory: (s_k, a_k, t_k)
            while not done:
                a = self.env.action_space.sample()
                real_trajectory.append((real_state, a, getattr(self.env, "elapsed_steps", 0)))
                real_state, _, done, _ = _step(self.env, a)

            start_idx = np.random.randint(
                low=0, high=max(1, len(real_trajectory) - max_rollout_length - 1)
            )
            model_state, _, t_k = real_trajectory[start_idx]
            try:
                self.env.set_state(model_state.tolist(), elapsed_steps=int(t_k))
            except TypeError:
                self.env.set_state(model_state.tolist())

            trajectory_errors = []
            for step in range(max_rollout_length):
                _, action_k, _ = real_trajectory[start_idx + step]
                next_model_state, _, _, _ = self.get_synthetic_transition(model_state, action_k)
                real_next_state, _, _ = real_trajectory[start_idx + step + 1]
                err = np.linalg.norm(real_next_state[:STATE_DIM] - next_model_state[:STATE_DIM])
                trajectory_errors.append(err)
                model_state = next_model_state

            all_errors.append(trajectory_errors)

        # Aggregate mean ± SEM
        max_T = max(len(traj) for traj in all_errors)
        err_mat = np.full((len(all_errors), max_T), np.nan, dtype=np.float32)
        for i, traj in enumerate(all_errors):
            err_mat[i, : len(traj)] = np.asarray(traj, dtype=np.float32)

        mean_err = np.nanmean(err_mat, axis=0)
        count = np.sum(~np.isnan(err_mat), axis=0).astype(np.float32)
        sem_err = np.nanstd(err_mat, axis=0, ddof=0) / np.sqrt(np.maximum(count, 1.0))
        steps = np.arange(max_T)

        plt.figure(figsize=(8, 6))
        plt.plot(steps, mean_err, linewidth=2, label="Mean error")
        plt.fill_between(steps, mean_err - sem_err, mean_err + sem_err, alpha=0.25, label="Std. error")
        plt.xlabel("Model rollout step")
        plt.ylabel("Prediction error (L2)")
        plt.title("Model prediction error across trajectories")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig("l2_error_rollout.png")


if __name__ == "__main__":
    env_name = "Pushing2D-v1"
    seeds = [0, 1, 2]
    start_timesteps = 5000
    eval_freq = 2500
    max_timesteps = 75000
    expl_noise = 0.1
    batch_size = 256
    num_nets = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_x = np.arange(1, (max_timesteps // eval_freq) + 1) * eval_freq

    # ---------- Q1.4.1: synthetic : real ratio ablation ----------
    ratios = [0.0, 0.2, 0.5, 0.8, 1.0]
    ratio_curves = {}
    for r in ratios:
        per_seed = []
        for sd in seeds:
            print(f"\n=== Ratio {r} | Seed {sd} ===")
            trainer = TrainerTD3(
                env_name, sd, start_timesteps, eval_freq, max_timesteps,
                expl_noise, batch_size, num_nets, device
            )
            _, succ = trainer.train_td3_with_mix(r)
            per_seed.append(succ)
        ratio_curves[r] = np.mean(np.array(per_seed, dtype=object), axis=0)

    plt.figure(figsize=(12, 8))
    for r, curve in ratio_curves.items():
        plt.plot(eval_x[:len(curve)], curve, label=f"Ratio = {r}")
    plt.title("Success Rate vs. Total Timesteps")
    plt.xlabel("Total Timesteps")
    plt.ylabel("Success Rate")
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("synthetic_ratio_success_rate.png")

    # ---------- Q1.4.2: rollout length ablation ----------
    lengths = [2, 4, 8]
    length_curves = {}
    for L in lengths:
        per_seed = []
        for sd in seeds:
            print(f"\n=== Rollout Length {L} | Seed {sd} ===")
            trainer = TrainerTD3(
                env_name, sd, start_timesteps, eval_freq, max_timesteps,
                expl_noise, batch_size, num_nets, device
            )
            _, succ = trainer.train_with_model_rollouts(L)
            per_seed.append(succ)
        length_curves[L] = np.mean(np.array(per_seed, dtype=object), axis=0)

    plt.figure(figsize=(12, 8))
    for L, curve in length_curves.items():
        plt.plot(eval_x[:len(curve)], curve, label=f"Rollout Length = {L}")
    plt.title("Success Rate vs. Total Timesteps")
    plt.xlabel("Total Timesteps")
    plt.ylabel("Success Rate")
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("rollout_length_success_rate.png")

    # ---------- Q1.4.3: model error vs rollout length ----------
    trainer = TrainerTD3(
        env_name, 0, start_timesteps, eval_freq, max_timesteps,
        expl_noise, batch_size, num_nets, device
    )
    trainer.compute_and_plot_model_error(max_rollout_length=15, num_trajectories=50)
