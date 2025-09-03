"""
Fitted Q Evaluation (FQE) for off-policy RL evaluation.
Estimates the expected return of a target policy using offline data.

Implementation follows Le et al. (2019):
  1. Train a Q-network on (s, a, r, s') tuples from offline dataset
  2. Use the Q-network to estimate V^π(s) = Q^π(s, π(s))
  3. Average over initial states to get estimated policy value
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class FQENetwork(nn.Module):
    """
    Simple Q-network for FQE.
    Maps (state, action) -> scalar Q-value.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list[int] = [256, 256]):
        super().__init__()
        layers = []
        prev_dim = state_dim + action_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class FQE:
    """
    Fitted Q Evaluation for estimating policy value from offline data.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int] = [256, 256],
        gamma: float = 0.99,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        epochs: int = 100,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        
        self.q_network = FQENetwork(state_dim, action_dim, hidden_dims).to(device)
        self.target_q_network = FQENetwork(state_dim, action_dim, hidden_dims).to(device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
    
    def fit(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        terminals: np.ndarray,
        policy,  # Callable that takes states and returns actions
    ) -> list[float]:
        """
        Train FQE Q-network on offline data.
        
        Args:
            observations: (N, state_dim)
            actions: (N, action_dim)
            rewards: (N,)
            next_observations: (N, state_dim)
            terminals: (N,) binary
            policy: Target policy to evaluate.
        
        Returns:
            List of training losses per epoch.
        """
        # Create dataset
        dataset = TensorDataset(
            torch.tensor(observations, dtype=torch.float32, device=self.device),
            torch.tensor(actions, dtype=torch.float32, device=self.device),
            torch.tensor(rewards, dtype=torch.float32, device=self.device),
            torch.tensor(next_observations, dtype=torch.float32, device=self.device),
            torch.tensor(terminals, dtype=torch.float32, device=self.device),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        losses = []
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            for batch_obs, batch_act, batch_rew, batch_next_obs, batch_term in loader:
                # Compute target: r + gamma * Q_target(s', pi(s')) * (1 - terminal)
                with torch.no_grad():
                    next_actions = policy(batch_next_obs)
                    next_q = self.target_q_network(batch_next_obs, next_actions).squeeze(-1)
                    target = batch_rew + self.gamma * next_q * (1.0 - batch_term)
                
                # Compute current Q
                current_q = self.q_network(batch_obs, batch_act).squeeze(-1)
                
                # MSE loss
                loss = self.criterion(current_q, target)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            losses.append(avg_loss)
            
            # Update target network periodically
            if (epoch + 1) % 10 == 0:
                self.target_q_network.load_state_dict(self.q_network.state_dict())
        
        return losses
    
    def estimate_value(
        self,
        initial_states: np.ndarray,
        policy,
    ) -> float:
        """
        Estimate the expected return of the policy from initial states.
        
        Args:
            initial_states: (N, state_dim) initial observation states.
            policy: Target policy callable.
        
        Returns:
            Estimated policy value (average discounted return).
        """
        self.q_network.eval()
        with torch.no_grad():
            states_t = torch.tensor(initial_states, dtype=torch.float32, device=self.device)
            actions_t = policy(states_t)
            q_values = self.q_network(states_t, actions_t).squeeze(-1)
            return float(q_values.mean().cpu().numpy())


def evaluate_policy_with_fqe(
    dataset,  # d3rlpy MDPDataset or similar
    policy,
    gamma: float = 0.99,
    epochs: int = 100,
    device: str = "cpu",
) -> float:
    """
    High-level helper to evaluate a d3rlpy policy with FQE.
    
    Args:
        dataset: d3rlpy dataset with observations, actions, rewards, terminals.
        policy: d3rlpy policy/agent.
        gamma: Discount factor.
        epochs: FQE training epochs.
        device: 'cpu' or 'cuda'.
    
    Returns:
        Estimated policy value.
    """
    observations = dataset.observations
    actions = dataset.actions
    rewards = dataset.rewards
    terminals = dataset.terminals
    
    # next_observations: shift observations by one step
    next_observations = np.roll(observations, shift=-1, axis=0)
    # Set next_obs for terminal states to 0 (or same as last)
    next_observations[terminals == 1] = observations[terminals == 1]
    
    state_dim = observations.shape[1]
    action_dim = actions.shape[1] if actions.ndim > 1 else 1
    
    fqe = FQE(
        state_dim=state_dim,
        action_dim=action_dim,
        gamma=gamma,
        epochs=epochs,
        device=device,
    )
    
    # Policy wrapper for FQE
    def policy_fn(states: torch.Tensor) -> torch.Tensor:
        states_np = states.cpu().numpy()
        actions_np = policy.predict(states_np)
        return torch.tensor(actions_np, dtype=torch.float32, device=device)
    
    print("Training FQE estimator...")
    fqe.fit(observations, actions, rewards, next_observations, terminals, policy_fn)
    
    # Estimate value from initial states (first 1000 states)
    initial_states = observations[:min(1000, len(observations))]
    value = fqe.estimate_value(initial_states, policy_fn)
    print(f"FQE estimated policy value: {value:.4f}")
    
    return value
