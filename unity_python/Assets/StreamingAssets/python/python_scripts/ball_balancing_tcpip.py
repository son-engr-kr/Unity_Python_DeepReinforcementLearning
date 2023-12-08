import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import namedtuple, deque
import torch.nn.functional as F
import itertools
import traceback
import os
# 신경망 정의
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.fc4 = nn.Linear(64, action_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)

# 경험 재생 메모리
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, state, action, next_state, reward):
        self.memory.append((state, action, next_state, reward))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# DQN 에이전트
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayMemory(10000)
        self.policy_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.batch_size = 128
        self.gamma = 0.99
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 1000
        self.steps_done = 0
        self.model_path = ""


        self.model_path = f"{os.path.dirname(__file__)}/../pytorch_models/ballbalancing_model_v4.pth"
        print(f"/inputEcho;modelPath;{self.model_path}", flush=True)

        if os.path.isfile(self.model_path):
            self.policy_net.load_state_dict(torch.load(self.model_path, map_location=device))
            self.policy_net.train()
            print(f"model loaded", flush=True)

        else:
            print(f";No saved model found. Starting with a new model.", flush=True)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
    def save_model(self):
        torch.save(self.policy_net.state_dict(), self.model_path)
        print(f"Model saved", flush=True)

    def select_action(self, state):
        sample = random.random()
        #Exploration : Exploitation
        #ratio decays over steps increase
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        np.exp(-1. * self.steps_done / self.eps_decay)
        # self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.action_size)]], device=device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                            if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

# 메인 학습 루프
def train(agent, episodes, state_size, action_size):
    try:
        for episode in range(episodes):
            print(f"ephisode {episode}")
            tcpip_protocol.send_request(episode)
            while True:
                with tcpip_protocol.state_lock:
                    if tcpip_protocol.ball_balancing_all_updated:
                        tcpip_protocol.ball_balancing_all_updated = False
                        # Unity로부터 초기 상태 및 보상 받기
                        state = [
                            tcpip_protocol.ball_balancing_state.BallPositionX,
                            tcpip_protocol.ball_balancing_state.BallPositionZ,

                            tcpip_protocol.ball_balancing_state.BallSpeedX,
                            tcpip_protocol.ball_balancing_state.BallSpeedZ,

                            tcpip_protocol.ball_balancing_state.PlateRX,
                            tcpip_protocol.ball_balancing_state.PlateRZ,

                            tcpip_protocol.ball_balancing_state.TargetPositionX,
                            tcpip_protocol.ball_balancing_state.TargetPositionZ,
                        ]

                        break
            state = torch.tensor([state], device=device, dtype=torch.float)
            for t in itertools.count():
                # print(f"/debugOutput: iter:{t}", flush=True)

                # 행동 선택 및 Unity로 보내기
                action = agent.select_action(state)
                tcpip_protocol.send_action(int(action.flatten().tolist()[0]))
                while True:
                    with tcpip_protocol.state_lock:
                        if tcpip_protocol.ball_balancing_all_updated:
                            tcpip_protocol.ball_balancing_all_updated = False
                            break
                # Unity로부터 다음 상태 및 보상 받기
                next_state = [
                    tcpip_protocol.ball_balancing_state.BallPositionX,
                    tcpip_protocol.ball_balancing_state.BallPositionZ,

                    tcpip_protocol.ball_balancing_state.BallSpeedX,
                    tcpip_protocol.ball_balancing_state.BallSpeedZ,

                    tcpip_protocol.ball_balancing_state.PlateRX,
                    tcpip_protocol.ball_balancing_state.PlateRZ,

                    tcpip_protocol.ball_balancing_state.TargetPositionX,
                    tcpip_protocol.ball_balancing_state.TargetPositionZ,
                ]
                reward = torch.tensor([tcpip_protocol.ball_balancing_reward], device=device)

                if not tcpip_protocol.ball_balancing_done:
                    next_state = torch.tensor([next_state], device=device, dtype=torch.float)
                else:
                    next_state = None

                # 메모리에 경험 저장
                agent.memory.push(state, action, next_state, reward)

                # 상태 업데이트
                state = next_state

                # 모델 최적화
                agent.optimize_model()
                if tcpip_protocol.ball_balancing_done:
                    print("episode done")
                    agent.steps_done+=1
                    break

            # 대상 신경망 업데이트
            if episode % TARGET_UPDATE == 0:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())
                agent.save_model()
    except Exception as e:
        tb = traceback.format_exc()
        print(f"/errorOutput;{tb}")

print(f"current script file dir:{os.path.dirname(__file__)}")
###===================server=====================
print("server start")
import tcpip_protocol

###===================server=====================

# Hyperparameters
EPISODES = 1000000
STATE_SIZE = 8  # 상태 크기 예시
ACTION_SIZE = 5  # 행동 크기 예시
TARGET_UPDATE = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 에이전트 초기화
agent = DQNAgent(STATE_SIZE, ACTION_SIZE)




# 학습 시작
train(agent, EPISODES, STATE_SIZE, ACTION_SIZE)
