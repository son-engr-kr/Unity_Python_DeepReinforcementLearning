import ball_balancing.ball_balancing_model as md
import ball_balancing.tcpip_protocol as protocol
import torch
import json
import os
import socket
import threading
import ctypes

print(f"current script file dir:{os.path.dirname(__file__)}")
###===================server=====================
print("server start")
###===================server=====================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





class TrainingClient:
    def __init__(self, client_socket):
        self.client_socket = client_socket
        self.state_size = 8
        self.action_size = 5
        self.agent = md.DQNAgent(self.state_size,
                                self.action_size,
                                torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        # self.lock = threading.Lock()
    def train_thread_start(self):
        client_thread = threading.Thread(target=self.train_loop, args=())
        client_thread.start()
        
    def train_loop(self):
        state_updated = False
        reward_updated = False
        done_updated = False

        need_request = True
        episode_count = 0
        try:
            while True:
                if need_request:
                    self.send_request(episode_count)
                    episode_count += 1
                    need_request = False
                header = self.read_header()
                if not header:
                    print("break")
                    break  # socket closed
                body_bytes = self.client_socket.recv(header.BODY_SIZE)
                data = json.loads(body_bytes)
                if header.SUBJECT_CODE == b"BB_STATE":
                    state_updated = True
                    state = protocol.BALL_BALANCING_STATE(**data)
                    state = torch.tensor([[
                                state.BallPositionX,
                                state.BallPositionZ,

                                state.BallSpeedX,
                                state.BallSpeedZ,

                                state.PlateRX,
                                state.PlateRZ,

                                state.TargetPositionX,
                                state.TargetPositionZ,
                            ]], device=device, dtype=torch.float)
                elif header.SUBJECT_CODE == b"BB_REWRD":
                    reward_updated = True
                    reward = torch.tensor(
                                [[protocol.BALL_BALANCING_REWARD(**data).Reward]],
                                device=device)
                elif header.SUBJECT_CODE == b"BB_DONE_":
                    done_updated = True
                    is_done = protocol.BALL_BALANCING_DONE(**data).Done
                    if is_done:
                        need_request = True
                if state_updated and reward_updated and done_updated:
                    state_updated = False
                    reward_updated = False
                    done_updated = False
                    action = self.agent.next_step(state, reward,is_done)
                    if is_done and episode_count%10 == 0:
                        self.agent.update_and_save_model()
                        print("model saved")
                    if action is not None:
                        self.send_action(int(action.flatten().tolist()[0]))

        finally:
            self.client_socket.close()
    @classmethod
    def to_json(cls, obj)->str:
        return json.dumps(obj.__dict__)
    @classmethod
    def from_json(cls, class_, json_str):
        data = json.loads(json_str)
        return class_(**data)
    def send_request(self, ep_count):
        body_bytes = json.dumps(protocol.BALL_BALANCING_NEXT_EPISODE_REQUIRE_TO_UNITY(ep_count).__dict__).encode(encoding = 'UTF-8', errors = 'strict')
        header_bytes = bytearray(protocol.HEADER(b"JSON",b"BB_EPRQS",len(body_bytes)))
        self.client_socket.sendall(header_bytes + body_bytes)
    def send_action(self, action:int):
        body_bytes = json.dumps(protocol.BALL_BALANCING_ACTION(action).__dict__).encode(encoding = 'UTF-8', errors = 'strict')
        header_bytes = bytearray(protocol.HEADER(b"JSON",b"BB_ACTN_",len(body_bytes)))
        self.client_socket.sendall(header_bytes + body_bytes)
    def read_header(self)->protocol.HEADER:
        byte_data = self.client_socket.recv(ctypes.sizeof(protocol.HEADER))
        if not byte_data:
            return None
        return protocol.HEADER.from_buffer_copy(byte_data)


host = 'localhost'
port = 11200

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((host, port))
print(f"Server start, host:{host}, port:{port}")

server_socket.listen()

while True:
    # 클라이언트의 연결을 기다림
    client_socket, address = server_socket.accept()
    training_client = TrainingClient(client_socket)
    training_client.train_thread_start()
    # 클라이언트 연결을 별도의 스레드에서 처리
    # client_thread = threading.Thread(target=handle_client_connection, args=(client_socket, address))
    # client_thread.start()

# 각 클라이언트 연결을 처리하기 위한 새 스레드 시작
