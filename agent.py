import numpy as np


class Agent:
    def __init__(self, env):
        self.env = env

    def sample(self, horizon, policy):
        """
        Sample a rollout from the agent.

        Arguments:
          horizon: (int) the length of the rollout
          policy: the policy that the agent will use for actions
        """
        rewards = []
        _out = self.env.reset()
        obs = _out[0] if (isinstance(_out, tuple) and len(_out) == 2) else _out
        states, actions, reward_sum, done = [obs], [], 0, False

        policy.reset()
        for t in range(horizon):
            actions.append(policy.act(states[t], t))
            step_out = self.env.step(actions[t])
            if isinstance(step_out, tuple) and len(step_out) == 5:
                state, reward, terminated, truncated, info = step_out
                done = bool(terminated or truncated)
            else:
                state, reward, done, info = step_out
            states.append(state)
            reward_sum += reward
            rewards.append(reward)
            if done:
                break

        # print("Rollout length: %d,\tTotal reward: %d,\t Last reward: %d" % (len(actions), reward_sum), reward)

        return {
            "obs": np.array(states),
            "ac": np.array(actions),
            "reward_sum": reward_sum,
            "rewards": np.array(rewards),
        }


class RandomPolicy:
    def __init__(self, action_dim):
        self.action_dim = action_dim

    def reset(self):
        pass

    def act(self, arg1, arg2):
        return np.random.uniform(size=self.action_dim) * 2 - 1


if __name__ == "__main__":
    import envs
    import gym

    env = gym.make("Pushing2D-v1")
    policy = RandomPolicy(2)
    agent = Agent(env)
    for _ in range(5):
        agent.sample(20, policy)
