import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import os
from collections import deque
from torch.utils.tensorboard import SummaryWriter

ROWS, COLS = 6, 7
WIN_LENGTH = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "dqn_puissance4.pth"
ONNX_PATH = "puissance4_model.onnx"

def opponent(player):
    return 2 if player == 1 else 1


class Puissance4Env:
    def __init__(self):
        self.board = np.zeros((ROWS, COLS), dtype=int)
        self.current_player = 1

    def reset(self):
        self.board = np.zeros((ROWS, COLS), dtype=int)
        self.current_player = 1
        if random.random() < 0.3:
            for _ in range(random.randint(1, 3)):
                self.play(random.choice(self.valid_actions()))
        return self.get_state()

    def get_state(self):
        state = np.zeros((3, ROWS, COLS), dtype=np.float32)
        state[0][self.board == 1] = 1
        state[1][self.board == 2] = 1
        state[2][self.board == 0] = 1
        return state

    def valid_actions(self):
        return [c for c in range(COLS) if self.board[0][c] == 0]

    def play(self, col):
        if col not in self.valid_actions():
            return False
        for row in reversed(range(ROWS)):
            if self.board[row][col] == 0:
                self.board[row][col] = self.current_player
                break
        self.current_player = opponent(self.current_player)
        return True

    def is_winning(self, player):
        for r in range(ROWS):
            for c in range(COLS - WIN_LENGTH + 1):
                if np.all(self.board[r, c:c+WIN_LENGTH] == player):
                    return True
        for r in range(ROWS - WIN_LENGTH + 1):
            for c in range(COLS):
                if np.all(self.board[r:r+WIN_LENGTH, c] == player):
                    return True
        for r in range(ROWS - WIN_LENGTH + 1):
            for c in range(COLS - WIN_LENGTH + 1):
                if all(self.board[r+i][c+i] == player for i in range(WIN_LENGTH)):
                    return True
        for r in range(WIN_LENGTH - 1, ROWS):
            for c in range(COLS - WIN_LENGTH + 1):
                if all(self.board[r-i][c+i] == player for i in range(WIN_LENGTH)):
                    return True
        return False

    def is_draw(self):
        return np.all(self.board[0] != 0)

    def game_over(self):
        return self.is_winning(1) or self.is_winning(2) or self.is_draw()


def compute_reward(env, player, action):
    opp = opponent(player)
    if env.is_winning(player):
        return 1.0
    if env.is_winning(opp):
        return -1.0
    if env.is_draw():
        return 0.0

    reward = -0.01
    if action == COLS // 2:
        reward += 0.05

    def count_patterns(length, target):
        count = 0
        for r in range(ROWS):
            for c in range(COLS - length + 1):
                window = env.board[r, c:c+length]
                if np.count_nonzero(window == target) == length and np.count_nonzero(window != 0) == length:
                    count += 1
        for r in range(ROWS - length + 1):
            for c in range(COLS):
                window = env.board[r:r+length, c]
                if np.count_nonzero(window == target) == length and np.count_nonzero(window != 0) == length:
                    count += 1
        for r in range(ROWS - length + 1):
            for c in range(COLS - length + 1):
                window = [env.board[r+i][c+i] for i in range(length)]
                if window.count(target) == length and 0 not in window:
                    count += 1
        for r in range(length - 1, ROWS):
            for c in range(COLS - length + 1):
                window = [env.board[r-i][c+i] for i in range(length)]
                if window.count(target) == length and 0 not in window:
                    count += 1
        return count

    reward += 0.05 * count_patterns(2, player)
    reward += 0.2 * count_patterns(3, player)
    reward -= 0.3 * count_patterns(3, opp)
    return reward


class ConvDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * ROWS * COLS, 128),
            nn.ReLU(),
            nn.Linear(128, COLS)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


def export_to_onnx(model):
    dummy_input = torch.randn(1, 3, ROWS, COLS).to(DEVICE)
    torch.onnx.export(model, dummy_input, ONNX_PATH, input_names=["input"], output_names=["output"], opset_version=11)


def train():
    writer = SummaryWriter()
    env = Puissance4Env()
    dqn = ConvDQN().to(DEVICE)
    target_dqn = ConvDQN().to(DEVICE)
    target_dqn.load_state_dict(dqn.state_dict())
    target_dqn.eval()

    optimizer = optim.Adam(dqn.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    replay_buffer = ReplayBuffer()

    batch_size = 64
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.99977
    epsilon_min = 0.05
    episodes = 100000
    warmup_steps = 1000
    best_winrate = 0
    win_history = deque(maxlen=100)

    # Warmup phase
    while len(replay_buffer) < warmup_steps:
        s = env.reset()
        done = False
        while not done:
            a = random.choice(env.valid_actions())
            env.play(a)
            ns = env.get_state()
            r = compute_reward(env, opponent(env.current_player), a)
            done = env.game_over()
            replay_buffer.push(s, a, r, ns, done)
            s = ns
            if not done:
                env.play(random.choice(env.valid_actions()))
                done = env.game_over()
                s = env.get_state()

    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        win = False

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            if random.random() < epsilon:
                action = random.choice(env.valid_actions())
            else:
                with torch.no_grad():
                    q_values = dqn(state_tensor)
                    for i in range(COLS):
                        if i not in env.valid_actions():
                            q_values[0][i] = -float("inf")
                    action = torch.argmax(q_values).item()

            env.play(action)
            next_state = env.get_state()
            reward = compute_reward(env, opponent(env.current_player), action)
            done = env.game_over()
            if reward > 0.5:
                win = True
            replay_buffer.push(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state

            if not done:
                with torch.no_grad():
                    q_values = target_dqn(torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0))
                    for i in range(COLS):
                        if i not in env.valid_actions():
                            q_values[0][i] = -float("inf")
                    action = torch.argmax(q_values).item()
                env.play(action)
                done = env.game_over()
                state = env.get_state()

        win_history.append(int(win))
        winrate = np.mean(win_history)

        if winrate > best_winrate:
            best_winrate = winrate
            torch.save(dqn.state_dict(), MODEL_PATH)
            export_to_onnx(dqn)

        batch = replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(np.array(states), dtype=torch.float32, device=DEVICE)
        actions = torch.tensor(actions, dtype=torch.long, device=DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=DEVICE)
        dones = [bool(d) for d in dones]
        dones = torch.tensor(dones, dtype=torch.bool, device=DEVICE)

        q_values = dqn(states)
        state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = dqn(next_states)
            for i, ns in enumerate(next_states):
                board = ns[0] - ns[1]
                env_mask = Puissance4Env()
                env_mask.board = board.cpu().numpy().astype(int)
                for j in range(COLS):
                    if j not in env_mask.valid_actions():
                        next_q[i][j] = -float("inf")
            next_actions = next_q.argmax(1)
            target_q = target_dqn(next_states)
            next_state_values = target_q.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            expected_values = rewards + gamma * next_state_values * (~dones)

        loss = criterion(state_action_values, expected_values)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(dqn.parameters(), max_norm=1.0)
        optimizer.step()

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        writer.add_scalar("Reward/Total", total_reward, ep)
        writer.add_scalar("Misc/Epsilon", epsilon, ep)
        writer.add_scalar("Loss", loss.item(), ep)
        writer.add_scalar("Performance/Winrate", winrate, ep)

        if ep % 10 == 0:
            target_dqn.load_state_dict(dqn.state_dict())

    writer.close()


if __name__ == "__main__":
    train()
