import glob

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

# The different trajectory counts to compare
N_values = [64, 128, 256]

# Collect reward statistics
reward_stats = {}

for N in N_values:
    reward_files = glob.glob(f"reward_{N}_seed_*.pt")
    all_rewards = []

    for reward_file in reward_files:
        rewards = torch.load(reward_file)
        all_rewards.append(np.array(rewards))

    if not all_rewards:
        continue  # skip if no data

    # Align runs by padding with NaNs (for unequal lengths)
    max_len = max(len(r) for r in all_rewards)
    padded_rewards = np.full((len(all_rewards), max_len), np.nan)

    for i, rewards in enumerate(all_rewards):
        padded_rewards[i, :len(rewards)] = rewards

    # Compute statistics across runs (ignoring NaNs)
    mean_rewards = np.nanmean(padded_rewards, axis=0)
    percentile_25 = np.nanpercentile(padded_rewards, 25, axis=0)
    percentile_75 = np.nanpercentile(padded_rewards, 75, axis=0)

    reward_stats[N] = {
        "mean": mean_rewards,
        "p25": percentile_25,
        "p75": percentile_75,
        "timesteps": np.arange(max_len),
    }

# --- Static Plot ---
plt.figure(figsize=(10, 6))
for N, stats in reward_stats.items():
    plt.plot(stats["timesteps"], stats["mean"], label=f"N={N}", linewidth=2)
    plt.fill_between(stats["timesteps"], stats["p25"], stats["p75"], alpha=0.2)

plt.xlabel("Time Step")
plt.ylabel("Reward")
legend = plt.legend(loc="lower right", frameon=True)
legend.get_frame().set_facecolor("white")
legend.get_frame().set_alpha(1.0)
plt.grid(True)
plt.tight_layout()
plt.savefig("reward_comparison.png", dpi=300)

# --- Animated Plot (mean reward curves growing over time) ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlabel("Time Step")
ax.set_ylabel("Reward")
ax.grid(True)

lines = {}
for N in reward_stats:
    (line,) = ax.plot([], [], lw=2, label=f"N={N}")
    lines[N] = line

legend = ax.legend(loc="lower right", frameon=True)
legend.get_frame().set_facecolor("white")
legend.get_frame().set_alpha(1.0)

max_len = max(len(stats["mean"]) for stats in reward_stats.values())
ax.set_xlim(0, max_len)
all_vals = np.concatenate([stats["mean"] for stats in reward_stats.values()])
ax.set_ylim(np.nanmin(all_vals), np.nanmax(all_vals))

def init():
    for line in lines.values():
        line.set_data([], [])
    return list(lines.values())

def update(frame):
    for N, stats in reward_stats.items():
        lines[N].set_data(stats["timesteps"][:frame], stats["mean"][:frame])
    return list(lines.values())

ani = animation.FuncAnimation(
    fig, update, frames=max_len,
    init_func=init, blit=True, interval=50, repeat=False
)

ani.save("reward_comparison.gif", writer="pillow", fps=30)
