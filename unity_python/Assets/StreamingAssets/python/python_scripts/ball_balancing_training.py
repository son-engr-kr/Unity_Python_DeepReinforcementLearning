from http import client
import ball_balancing.ball_balancing_model as md
import ball_balancing.tcpip_protocol as protocol
import torch
import json
import os
import socket
import threading
import ctypes
import time
print(f"current script file dir:{os.path.dirname(__file__)}")
###===================server=====================
print("server start")
###===================server=====================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class TrainingClient:
    client_list = []
    local_train_done_count = 0
    lock = threading.Lock()

    state_size = 8
    action_size = 5
    global_net_opt_package = None
    model_path = ""
    device = None
    
    def __init__(self, client_socket, client_id):
        self.client_socket = client_socket
        # print(f"TrainingClient init: {TrainingClient.global_net_opt_package}")
        self.agent = md.DQNAgent(TrainingClient.state_size,
                                TrainingClient.action_size,
                                TrainingClient.device,
                                TrainingClient.global_net_opt_package)
        with TrainingClient.lock:
            TrainingClient.client_list.append(self)
        self.wait_for_optimize = False
        self.client_id = client_id
    @classmethod
    def global_optimize_and_save_model(cls):
        cls.local_train_done_count = 0
        for client in TrainingClient.client_list:
            with TrainingClient.lock:
                # print(f"memory move(bf): {len(client.agent.local_memory)} -> {len(TrainingClient.global_net_opt_package.memory)}")
                client.agent.local_memory.move_to_other(TrainingClient.global_net_opt_package.memory)
                # print(f"memory move(af): {len(client.agent.local_memory)} -> {len(TrainingClient.global_net_opt_package.memory)}")
        with TrainingClient.lock:
            for idx in range(1):
                cls.global_net_opt_package.optimize()
        time.sleep(0.3)
        for client in TrainingClient.client_list:
            client.agent.update_local_model()
            with TrainingClient.lock:
                client.wait_for_optimize = False
        ##save model
        with TrainingClient.lock:
            torch.save(cls.global_net_opt_package.net.state_dict(), cls.model_path)
            print("model saved!")
    def train_thread_start(self):
        client_thread = threading.Thread(target=self.train_loop, args=())
        client_thread.start()
        
    def train_loop(self):
        time.sleep(3)
        print("train start")
        state_updated = False
        reward_updated = False
        done_updated = False

        need_request = True
        episode_count = 0
        try:
            while True:
                if need_request:
                    # print("next episode")
                    self.send_request(episode_count)
                    episode_count += 1
                    need_request = False
                    # print(f"client-{self.client_id}:request done-ep[{episode_count}]")
                # print(f"client-{self.client_id}:try to read header")
                
                header = self.read_header()
                # print(f"client-{self.client_id}:read header: {header.SUBJECT_CODE}")

                if not header:
                    print(f"client-{self.client_id}:tcp done")
                    with TrainingClient.lock:
                        TrainingClient.client_list.remove(self)
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
                    # print(f"client-{self.client_id}:receive done:{is_done}")

                    
                if state_updated and reward_updated and done_updated:
                    if is_done:
                        need_request = True
                        # print(f"client-{self.client_id}:done")
                    # time.sleep(1.234)
                    # print(f"client-{self.client_id}:step")

                    state_updated = False
                    reward_updated = False
                    done_updated = False
                    action = self.agent.next_step(state, reward,is_done)
                    if action is not None:#여기서 send_action을 안보내니 다음 반응도 전부 없었던 것임. 이제 None일 리 없으니 잘 됨
                        self.send_action(int(action.flatten().tolist()[0]))
                    if is_done and episode_count%5 == 0:
                        optimize_flag = False
                        with TrainingClient.lock:
                            self.wait_for_optimize = True
                            TrainingClient.local_train_done_count += 1
                            if len(TrainingClient.client_list) == TrainingClient.local_train_done_count:
                                optimize_flag = True
                        if optimize_flag:
                            TrainingClient.global_optimize_and_save_model()
                        while True:
                            time.sleep(0.3)
                            with TrainingClient.lock:
                                if not self.wait_for_optimize:
                                    # print(f"client-{self.client_id}:optimize done")
                                    break
        except Exception as e:
            print(f"error|client-{self.client_id}:{e}")            

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

if __name__ == "__main__":

    host = 'localhost'
    port = 11200

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    print(f"Server start, host:{host}, port:{port}")

    server_socket.listen()
    TrainingClient.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TrainingClient.model_path = f"{os.path.dirname(__file__)}/../pytorch_models/ballbalancing_model_v5_3.pth"
    TrainingClient.global_net_opt_package = md.NetOptimizerPackage(TrainingClient.state_size,TrainingClient.action_size,TrainingClient.device)
    if os.path.isfile(TrainingClient.model_path):
        TrainingClient.global_net_opt_package.net.load_state_dict(torch.load(TrainingClient.model_path, map_location=TrainingClient.device))
        TrainingClient.global_net_opt_package.net.train()
        print(f"model loaded", flush=True)

    else:
        print(f";No saved model found. Starting with a new model.", flush=True)

    client_count = 0
    while True:
        # 클라이언트의 연결을 기다림
        client_socket, address = server_socket.accept()
        training_client = TrainingClient(client_socket, client_count)
        training_client.train_thread_start()
        client_count += 1
        print(f"new client: {client_count}")
        # 클라이언트 연결을 별도의 스레드에서 처리
        # client_thread = threading.Thread(target=handle_client_connection, args=(client_socket, address))
        # client_thread.start()

    # 각 클라이언트 연결을 처리하기 위한 새 스레드 시작
