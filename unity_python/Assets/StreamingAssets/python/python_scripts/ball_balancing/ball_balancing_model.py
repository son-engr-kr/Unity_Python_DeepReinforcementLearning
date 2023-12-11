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
class DQN(nn.Module):##v5.1
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
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
    def get_at_index(self, idx):
        return self.memory[idx]
    def sample(self, batch_size):
        indices = random.sample(range(len(self.memory)), batch_size)
        transitions = [self.memory[idx] for idx in indices]
        return indices, transitions
    def __getitem__(self, idx):
        return self.memory[idx]
    def __len__(self):
        return len(self.memory)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# DQN 에이전트
class DQNAgent:
    def __init__(self, state_size, action_size, device):
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
        self.eps_decay = 100
        self.steps_done = 0
        self.model_path = ""

        self.device = device
        
        ##======for training ======
        self.state = None
        ##=========================

        self.model_path = f"{os.path.dirname(__file__)}/../../pytorch_models/ballbalancing_model_v5_2.pth"
        print(f"/inputEcho;modelPath;{self.model_path}", flush=True)

        if os.path.isfile(self.model_path):
            self.policy_net.load_state_dict(torch.load(self.model_path, map_location=self.device))
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
            return torch.tensor([[random.randrange(self.action_size)]], device=self.device, dtype=torch.long)

    def compute_multistep_return(self, current_index, n_steps, gamma):
        return_value = 0.0
        for i in range(n_steps):
            step_index = (current_index + i) % len(self.memory)
            state, action, next_state, reward = self.memory[step_index]
            return_value += (gamma ** i) * reward
            if next_state is None:  # 에피소드가 종료된 경우
                break  # 더 이상 미래 보상을 계산하지 않음
        return return_value
    def update_and_save_model(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.save_model()
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        '''
        transitions = [
            (state0, action0, next_state0, reward0),
            (state1, action1, next_state1, reward1),
            (state2, action2, next_state2, reward2)
        ]
        '''
        indices, transitions = self.memory.sample(self.batch_size)
        '''
        Transition(
            state=(state0, state1, state2),
            action=(action0, action1, action2),
            next_state=(next_state0, next_state1(None), next_state2),
            reward=(reward0, reward1, reward2)
        )
        '''
        batch = Transition(*zip(*transitions))
        #tensor([True, False, True]) (next_state가 None이면 False)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        # tensor([next_state0, next_state2]) - dim[2,input_len]
        non_final_next_states = torch.cat([s for s in batch.next_state
                                            if s is not None])
        '''
        batch_size가 128일때
        state_batch = torch.cat([state0, state1, state2]) - dim[128*input_len]
        action_batch = torch.cat([action0, action1, action2]) - dim[128 * action_len(1)]
        reward_batch = torch.cat([reward0, reward1, reward2]) - dim[128 * reward_len(1)]
        '''                                            
                                            
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        # reward_batch = torch.cat(batch.reward) # one-step

        multistep_returns = torch.tensor([self.compute_multistep_return(idx, 10, self.gamma)
                                  for idx in indices], device=self.device)#multi-step

        # dim[128, action_size(5)]
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        # dim[128]
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        expected_state_action_values = multistep_returns

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
    def next_step(self, next_state, reward, is_done):
        if self.state is None:
            self.state = next_state
            return None
        else:
            # print(f"/debugOutput: iter:{t}", flush=True)

            # 행동 선택 및 Unity로 보내기
            action = self.select_action(self.state)

            if not is_done:
                pass
                # next_state = torch.tensor([next_state], device=device, dtype=torch.float)
            else:
                next_state = None

            # 메모리에 경험 저장
            self.memory.push(self.state, action, next_state, reward)

            # 상태 업데이트
            self.state = next_state

            # 모델 최적화
            self.optimize_model()
            if is_done:
                print("episode done")
                self.steps_done+=1
            return action
