import copy
import glob
import os
import time

import colorednoise
import torch
import warp as wp

# from tqdm import tqdm
from mineral.agents.agent import Agent


class CEMMPCAgent(Agent):
    r"""Cross Entropy Method Model Predictive Control."""

    def __init__(self, full_cfg, **kwargs):
        self.network_config = full_cfg.agent.network
        self.params = full_cfg.agent.params
        self.num_actors = self.params.num_actors
        self.max_agent_steps = int(self.params.max_agent_steps)
        self.render_results = self.params.render_results
        self.seed = self.params.seed

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

        # CEM with memory
        self.previous_elite_actions = None

    def clone_state(self, state):
        """Copy the all attributes of a state."""
        # Create a new empty State object of the same type
        s = type(state)()

        # Attributes to clone with wp.clone
        wp_clone_attrs = [
            "body_f", "body_q", "body_qd", "joint_q", "joint_qd", "mpm_C", "mpm_F", "mpm_F_trial",
            "mpm_grid_m", "mpm_grid_mv", "mpm_grid_v", "mpm_stress", "mpm_x", "mpm_v"
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
            # Specify N values to evaluate
            # e.g. N_values = [100, 200, 300] or single N_values = [100]
            # self.eval_multiple([100, 200, 300])
            # self.eval_multiple([64, 128, 256])
            self.eval_multiple([128])


    def generate_action_sequences(self, mean, std, elite_actions, iteration=0):
        mean_exp = mean.unsqueeze(0).expand(self.N, -1, -1)     # Shape: (N, H, action_dim)
        std_exp = std.unsqueeze(0).expand(self.N, -1, -1)       # Shape: (N, H, action_dim)

        # Generate colored noise (N, action_dim, H) --> transpose to (N, H, action_dim)
        # Colored noise generated should be temporally correlated along the axis of H
        noise = colorednoise.powerlaw_psd_gaussian(
            self.beta, size=(self.N, self.action_dim, self.H)
        ).transpose(0, 2, 1)
        noise = torch.from_numpy(noise).cuda().float()          # Shape: (N, H, action_dim)

        # Scale the distribution by noise * std
        action_sequences = mean_exp + noise * std_exp           # Shape: (N, H, action_dim)

        # [iCEM 3.3] - Clipping at the action boundaries (clip)
        # Clip actions to range of [-1, 1]
        env_action_low = torch.from_numpy(self.env.action_space.low).float().to(self.device)    # low:  -1
        env_action_high = torch.from_numpy(self.env.action_space.high).float().to(self.device)  # high:  1
        action_sequences = torch.clip(action_sequences, env_action_low, env_action_high)

        # [iCEM 3.2] - CEM with memory
        # Keep a fraction of the elites and shift elites every iteration
        if (iteration == 0) and elite_actions is not None:
            num_elite_to_add = int(self.keep_elite_fraction * self.K)
            shifted_elite = torch.roll(elite_actions[:num_elite_to_add], -1, dims=1)
            action_sequences[:num_elite_to_add] = shifted_elite

        elif elite_actions is not None:
            num_elite_to_add = int(self.keep_elite_fraction * self.K)
            action_sequences[:num_elite_to_add] = elite_actions[:num_elite_to_add]

        if iteration == self.iterations - 1:
            action_sequences[-1] = mean

        return action_sequences

    def evaluate_action_sequence_batch(self, init_state, action_sequences):
        # Initialise rewards tensor with shape (N,)
        batch_size = self.num_actors
        rewards = torch.zeros(self.N, device=self.device)

        # Batch the rewards based on the amount of samples
        for start_idx in range(0, self.N, batch_size):
            end_idx = min(start_idx + batch_size, self.N)
            batch_sequences = action_sequences[start_idx:end_idx]
            batch_N = batch_sequences.shape[0]

            # Copy the current state of the env to perform actions
            self.env.state_0 = self.clone_state(init_state)
            batch_rewards = torch.zeros(batch_N, device=self.device)

            for h in range(self.H):
                actions_h = batch_sequences[:, h, :]   # Shape: (batch_N, action_dim)

                # Pad actions to match num_actors if needed
                if batch_N < self.num_actors:
                    # Repeat the last action or pad with zeros
                    padded_actions = torch.zeros(self.num_actors, self.action_dim, device=self.device)
                    padded_actions[:batch_N] = actions_h
                    actions_h = padded_actions

                _, r ,_, _ = self.env.step(actions_h)
                batch_rewards += r[:batch_N]

            rewards[start_idx:end_idx] = batch_rewards

        return rewards

    def cem_plan(self, init_state, previous_elite_actions=None):
        mean = torch.zeros((self.H, self.action_dim), device=self.device)       # Mean
        std = torch.ones((self.H, self.action_dim), device=self.device) * 0.5   # Standard deviation

        prev_mean = mean.clone()
        prev_std = std.clone()

        # Use previous elite actions from last timestep for first iteration
        elite_actions = previous_elite_actions

        for iteration in range(self.iterations):
            # Generation action sequences and allocate rewards for each action sequence
            action_sequences = self.generate_action_sequences(mean, std, elite_actions, iteration)
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
        return best_action_sequence[0], elite_actions

    def run_cem_mpc(self, save_name="trajectory.pt"):
        # Initialise environment
        obs = self.env.reset()
        self.obs = self._convert_obs(obs)
        self.dones = torch.zeros((self.num_actors,), dtype=torch.bool, device=self.device)
        total_reward = 0.0

        # Create list to save actions
        best_actions_list = []

        # Main loop
        print(f"\n===== Running evaluation with N={self.N} =====")
        for timestep in range(self.timesteps):
            if self.dones[0]:
                break  # Stop if real environment is done

            # Save the initial state
            init_state = self.clone_state(self.env.state_0)

            # Evaluate new best action
            best_action, elite_actions = self.cem_plan(init_state, self.previous_elite_actions)
            actions = best_action.unsqueeze(0).repeat(self.num_actors, 1)   # Shape: (64, action_dim)

            # Store elite actions for next timestep
            self.previous_elite_actions = elite_actions

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
            # tqdm.write(f"Timestep {timestep + 1} | Action: {best_action} | Reward: {reward[0]:.3f}")

        print("Evaluation complete")
        print(f"Total reward: {total_reward:.3f}")

        # Save trajectory as PyTorch file
        best_actions_tensor = torch.stack(best_actions_list)

        torch.save(best_actions_tensor, save_name)
        print(f"Trajectory {self.N} saved")

    def eval_multiple(self, N_values):
        for N in N_values:
            # Override N and reset previous_elite_actions
            self.N = N
            self.previous_elite_actions = None

            start_time = time.time()
            self.run_cem_mpc(save_name=f"trajectory_{N}_seed_{self.seed}.pt")
            end_time = time.time()
            print(f"Time taken: {end_time - start_time}")


    def replay_trajectory(self):
        # Initialise environment
        obs = self.env.reset()
        self.obs = self._convert_obs(obs)

        # Collect all trajectory files
        trajectory_files = glob.glob(f"trajectory_*_seed_{self.seed}.pt")
        for file in trajectory_files:
            actions_to_replay = torch.load(file)

            rewards = []
            timestep = 0

            for action in actions_to_replay:
                obs, reward, done, _ = self.env.step(action)
                self.obs = self._convert_obs(obs)
                self.dones = done

                print(f"Timestep {timestep + 1} | Action: {action} | Reward: {reward[0]:.3f}")
                timestep += 1
                rewards.append(reward[0].item())
            print("Trajectory completed")

            # Save rewards tensor as pytorch file
            base_name = os.path.basename(file)
            idx = base_name.replace("trajectory_", "").replace(".pt", "")
            reward_filename = f"reward_{idx}.pt"

            torch.save(rewards, reward_filename)
