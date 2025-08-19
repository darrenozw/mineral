import copy
import time

import colorednoise
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import warp as wp

from mineral.agents.agent import Agent


class CEMMPCAgent(Agent):
    r"""Cross Entropy Method Model Predictive Control."""

    def __init__(self, full_cfg, **kwargs):
        self.network_config = full_cfg.agent.network
        self.params = full_cfg.agent.params
        self.num_actors = self.params.num_actors
        self.max_agent_steps = int(self.params.max_agent_steps)
        self.render_results = self.params.render_results

        # Collect all CEM MPC Parameters
        self.cem_mpc_params = full_cfg.agent.cem_mpc_params
        self.H = self.cem_mpc_params.H
        self.N = self.cem_mpc_params.N
        self.K = self.cem_mpc_params.K
        self.iterations = self.cem_mpc_params.iterations
        self.timesteps = self.cem_mpc_params.timesteps
        self.beta = self.cem_mpc_params.beta
        # self.gamma = self.cem_mpc_params.gamma
        self.keep_elite_fraction = self.cem_mpc_params.keep_elite_fraction
        self.alpha = self.cem_mpc_params.alpha

        super().__init__(full_cfg, **kwargs)

        self.obs = None
        self.dones = None

    def clone_state(self, state):
        s = type(state)()  # create a new empty State object of the same type

        # Attributes to clone with wp.clone
        wp_clone_attrs = [
            "body_f", "body_q", "body_qd", "joint_q", "joint_qd", "mpm_C", "mpm_F", "mpm_F_trial",
            "mpm_grid_m", "mpm_grid_mv", "mpm_grid_v", "mpm_stress", "mpm_x", "mpm_v",
        ]

        # Attributes to deep copy
        deepcopy_attrs = [ "particle_f", "particle_q", "particle_qd"]

        for attr in wp_clone_attrs:
            if hasattr(state, attr):
                setattr(s, attr, wp.clone(getattr(state, attr)))
        for attr in deepcopy_attrs:
            if hasattr(state, attr):
                setattr(s, attr, copy.deepcopy(getattr(state, attr)))

        return s


    def eval(self):
        # render_results
        # True: use saved trajectory and render animation
        # False: perform evaluation with CEM MPC
        if self.render_results:
            self.replay_trajectory()

        else:
            start_time = time.time()
            self.run_cem_mpc()
            end_time = time.time()
            print(f"Time taken: {end_time - start_time}")

    def generate_action_sequences(self, mean, std, elite_actions, iteration=0):
        mean_exp = mean.unsqueeze(0).expand(self.N, -1, -1)     # Shape: (N, H, action_dim)
        std_exp = std.unsqueeze(0).expand(self.N, -1, -1)       # Shape: (N, H, action_dim)
        action_sequences = torch.normal(mean_exp, std_exp)      # Shape: (N, H, action_dim)

        # Generate colored noise for each action dimension
        noise = torch.zeros((self.N, self.H, self.action_dim), device=self.device)
        for d in range(self.action_dim):
            # iCEM 3.1 - Colored noise and correlations
            # S(f) = 1 / (f^beta)
            noise_np = colorednoise.powerlaw_psd_gaussian(self.beta, (self.N, self.H))
            noise[:, :, d] = torch.from_numpy(noise_np).cuda()

        # Add the colored noise to the samples
        action_sequences += noise * std_exp

        # [iCEM 3.3] - Clipping at the action boundaries (clip)
        # Clip actions to range of [-1, 1]
        env_action_low = torch.from_numpy(self.env.action_space.low).float().to(self.device)    # low:  -1
        env_action_high = torch.from_numpy(self.env.action_space.high).float().to(self.device)  # high:  1
        action_sequences = torch.clip(action_sequences, env_action_low, env_action_high)

        # [iCEM 3.2] - CEM with memory
        # Keep a fraction of the elites and shift elites every iteration
        if (iteration == 0) and elite_actions is not None:
            num_elite_to_add = int(self.keep_elite_fraction * self.K)
            shifted_elite = np.roll(elite_actions[:num_elite_to_add], -1, axis=1)
            action_sequences[:num_elite_to_add] = shifted_elite

        elif elite_actions is not None:
            num_elite_to_add = int(self.keep_elite_fraction * self.K)
            action_sequences[:num_elite_to_add] = elite_actions[:num_elite_to_add]

        if (iteration == self.iterations - 1):
            action_sequences[-1] = mean


        # if iteration == 0:


        return action_sequences

    def evaluate_action_sequence_batch(self, init_state, action_sequences):
        # Initialise rewards tensor with shape (N,)
        rewards = torch.zeros(self.N, device=self.device)

        # Copy the current state of the env to perform actions
        self.env.state_0 = self.clone_state(init_state)

        # Evaluating the actions for each time step in the horizon H
        for h in range(self.H):
            actions_h = action_sequences[:, h, :]   # Shape: (N, action_dim)
            _, r ,_, _ = self.env.step(actions_h)
            rewards += r

        return rewards

    def cem_plan(self, init_state):
        mean = torch.zeros((self.H, self.action_dim), device=self.device)       # Mean
        std = torch.ones((self.H, self.action_dim), device=self.device) * 0.5   # Standard deviation

        prev_mean = mean.clone()
        prev_std = std.clone()
        elite_actions = None

        for iteration in range(self.iterations):
            # Generation action sequences
            action_sequences = self.generate_action_sequences(mean, std, elite_actions, iteration)

            # Allocate rewards for each action sequence
            rewards = self.evaluate_action_sequence_batch(init_state, action_sequences)

            # Sort the rewards array and pick only the top K elements
            _, elite_idxs = torch.topk(rewards, self.K)
            elite_actions = action_sequences[elite_idxs]

            # [iCEM 3.3] - Executing the best action (best-a)
            # Identify the best trajectory among elites (first element in sorted elites)
            best_action_sequence = action_sequences[elite_idxs[0]]

            # Update the values of mu and sigma after iteration
            new_mean = elite_actions.mean(dim=0)
            new_std = elite_actions.std(dim=0) + 1e-3   # Stability

            # Momentum update
            mean = (self.alpha * prev_mean) + ((1 - self.alpha) * new_mean)
            std = (self.alpha * prev_std) + ((1 - self.alpha) * new_std)

            # Update values for next iteration
            prev_mean = mean.clone()
            prev_std = std.clone()

        # Return the first action of the best sequence
        return best_action_sequence[0]

    def run_cem_mpc(self):
        # Initialise environment
        obs = self.env.reset()
        self.obs = self._convert_obs(obs)
        self.dones = torch.zeros((self.num_actors,), dtype=torch.bool, device=self.device)
        total_reward = 0.0

        # Create list to save actions
        best_actions_list = []

        # Main loop
        for timestep in range(self.timesteps):
            if self.dones[0]:
                break  # Stop if real environment is done

            # Save the initial state
            init_state = self.clone_state(self.env.state_0)

            # Evaluate new best action
            best_action = self.cem_plan(init_state)
            actions = best_action.unsqueeze(0).repeat(self.N, 1)    # Shape: (N, action_dim)

            # Reset state of all environments
            self.env.state_0 = init_state

            # Step the environment forward
            obs, reward, done, _ = self.env.step(actions)
            self.obs = self._convert_obs(obs)
            self.dones = done
            total_reward += reward[0]

            # Append actions to best action list
            best_actions_list.append(best_action.clone())

            print(f"Timestep {timestep + 1} | Action: {best_action} | Reward: {reward[0]:.3f}")

        print("Evaluation complete")
        print(f"Total reward: {total_reward:.3f}")

        # Save trajectory as PyTorch file
        best_actions_tensor = torch.stack(best_actions_list)
        torch.save(best_actions_tensor, 'trajectory.pt')
        print("Trajectory saved")

    def replay_trajectory(self):
        # Initialise environment
        obs = self.env.reset()
        self.obs = self._convert_obs(obs)
        self.dones = torch.zeros((self.num_actors,), dtype=torch.bool, device=self.device)

        rewards = []
        actions_to_replay = torch.load('trajectory.pt')

        timestep = 0
        for action in actions_to_replay:
            obs, reward, done, _ = self.env.step(action)
            self.obs = self._convert_obs(obs)
            self.dones = done

            print(f"Timestep {timestep + 1} | Action: {action} | Reward: {reward[0]:.3f}")
            timestep += 1
            rewards.append(reward[0].item())
        print("Trajectory completed")

        plt.figure(figsize=(10, 6))
        plt.plot(rewards)
        plt.xlabel("Timestep")
        plt.ylabel("Reward")
        plt.title("CEM MPC on Rewarped RollingFlat Task")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("reward_plot.png", dpi=300)

        # --- Animated Plot ---
        fig, ax = plt.subplots()
        ax.set_xlim(0, len(rewards))
        ax.set_ylim(min(rewards) - 0.1, max(rewards) + 0.1)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Reward")
        ax.set_title("CEM MPC on Rewarped RollingFlat Task")

        line, = ax.plot([], [], lw=2, label="Step Reward")
        ax.legend()

        xdata, ydata = [], []

        def init():
            line.set_data([], [])
            return line,

        def update(frame):
            xdata.append(frame)
            ydata.append(rewards[frame])
            line.set_data(xdata, ydata)
            return line,

        ani = animation.FuncAnimation(
            fig, update, frames=len(rewards),
            init_func=init, blit=True, interval=100, repeat=False
        )

        # Save animation as GIF
        ani.save("reward_animation.gif", writer="pillow", fps=30)
