import importlib, sys
for mod in ["rover_env", "ppo_agent"]:
    if mod in sys.modules:
        importlib.reload(sys.modules[mod])

import numpy as np
import pickle
from rover_env import RoverEnv
from ppo_agent import PPOAgent

env        = RoverEnv()
probe      = env.reset()
state_size = len(probe)          # 17
action_size = 4
print(f"[PPO] State size: {state_size}")

agent    = PPOAgent(state_size, action_size)
episodes = 1500

# PPO collects a rollout horizon then updates — no replay buffer needed
ROLLOUT_STEPS = 512    # collect this many env steps before each PPO update

ppo_rewards = []
ppo_success = []
ppo_battery = []
ppo_margins = []
ppo_steps   = []

global_step  = 0
state        = env.reset()
ep_reward    = 0
ep_margins   = []
reached_goal = False
ep_count     = 0

print("[PPO] Training started...\n")

while ep_count < episodes:
    # ── Collect rollout ────────────────────────────────────────────────────────
    for _ in range(ROLLOUT_STEPS):
        action, log_prob, value = agent.act(state)
        next_state, reward, done = env.step(action)

        ep_margins.append(next_state[6])
        agent.store(state, action, log_prob, reward, value, done)

        state        = next_state
        ep_reward   += reward
        global_step += 1

        if done:
            if reward >= 100:
                reached_goal = True

            ppo_rewards.append(ep_reward)
            ppo_success.append(1 if reached_goal else 0)
            ppo_battery.append(env.battery)
            ppo_margins.append(float(np.mean(ep_margins)))
            ppo_steps.append(env.steps)

            sr = (np.mean(ppo_success[-100:]) if len(ppo_success) >= 100
                  else np.mean(ppo_success))
            print(f"[PPO] Ep {ep_count+1:4d}/{episodes} | "
                  f"R: {ep_reward:7.1f} | SR(100): {sr:.2f} | "
                  f"Bat: {env.battery} | Steps: {global_step}")

            ep_count    += 1
            ep_reward    = 0
            ep_margins   = []
            reached_goal = False
            state        = env.reset()

            if ep_count >= episodes:
                break

    # ── PPO update on collected rollout ───────────────────────────────────────
    if len(agent.states) > 0:
        _, _, next_val = agent.act(state)
        import torch
        next_value = next_val.item() if not done else 0.0
        agent.update(next_value)

# Save results
results = {
    "rewards" : ppo_rewards,
    "success" : ppo_success,
    "battery" : ppo_battery,
    "margins" : ppo_margins,
    "steps"   : ppo_steps
}
with open("ppo_results.pkl", "wb") as f:
    pickle.dump(results, f)

print(f"\n[PPO] Final SR (last 100): {np.mean(ppo_success[-100:]):.2f}")
print(f"[PPO] Avg battery left   : {np.mean(ppo_battery[-100:]):.1f}")
print("[PPO] Results saved to ppo_results.pkl")
