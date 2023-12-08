import socket, time
import marshal
import ctypes
import threading
import json
class HEADER(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("DATA_TYPE",ctypes.c_char * 4),
        ("SUBJECT_CODE",ctypes.c_char * 8),
        ("BODY_SIZE",ctypes.c_int32)]
##=========JSON Packet=============
class BALL_BALANCING_STATE:
    def __init__(self, 
                BallPositionX=0,
                BallPositionZ=0,
                BallSpeedX=0,
                BallSpeedZ=0,
                PlateRX=0,
                PlateRZ=0,
                TargetPositionX=0,
                TargetPositionZ=0
                ):
        self.BallPositionX:float=BallPositionX
        self.BallPositionZ:float=BallPositionZ

        self.BallSpeedX:float=BallSpeedX
        self.BallSpeedZ:float=BallSpeedZ

        self.PlateRX:float=PlateRX
        self.PlateRZ:float=PlateRZ

        self.TargetPositionX:float=TargetPositionX
        self.TargetPositionZ:float=TargetPositionZ
class BALL_BALANCING_REWARD:
    def __init__(self, Reward=0):
        self.Reward:float = Reward
class BALL_BALANCING_DONE:
    def __init__(self, Done=False):
        self.Done:bool = Done

class BALL_BALANCING_ACTION:
    def __init__(self, Action = 0):
        self.Action:int = Action

class BALL_BALANCING_NEXT_EPISODE_REQUIRE_TO_UNITY:
    def __init__(self, EPISODE_COUNT = 0):
        self.EPISODE_COUNT = EPISODE_COUNT
##!=========JSON Packet=============



def to_json(obj)->str:
    return json.dumps(obj.__dict__)
def from_json(class_, json_str):
    data = json.loads(json_str)
    return class_(**data)

def send_request(ep_count):
    body_bytes = json.dumps(BALL_BALANCING_NEXT_EPISODE_REQUIRE_TO_UNITY(ep_count).__dict__).encode(encoding = 'UTF-8', errors = 'strict')
    header_bytes = bytearray(HEADER(b"JSON",b"BB_EPRQS",len(body_bytes)))
    current_client.sendall(header_bytes + body_bytes)
def send_action(action:int):
    body_bytes = json.dumps(BALL_BALANCING_ACTION(action).__dict__).encode(encoding = 'UTF-8', errors = 'strict')
    header_bytes = bytearray(HEADER(b"JSON",b"BB_ACTN_",len(body_bytes)))
    current_client.sendall(header_bytes + body_bytes)
def read_header(client_socket)->HEADER:
    byte_data = client_socket.recv(ctypes.sizeof(HEADER))
    if not byte_data:
        return None
    return HEADER.from_buffer_copy(byte_data)

# def get_state()->BALL_BALANCING_STATE:
#     return ball_balancing_state
# def get_reward()->float:
#     return ball_balancing_reward
# def get_done_flag()->bool:
#     return ball_balancing_done
state_lock = threading.Lock()
ball_balancing_all_updated = False
ball_balancing_state:BALL_BALANCING_STATE = BALL_BALANCING_STATE(0,0, 0,0, 0,0, 0,0)
ball_balancing_reward:float = 0
ball_balancing_done:bool = False
def handle_client_connection(client_socket):
    global ball_balancing_all_updated
    global ball_balancing_state
    global ball_balancing_reward
    global ball_balancing_done
    
    state_updated = False
    reward_updated = False
    done_updated = False
    try:
        while True:
            #receive header
            header = read_header(client_socket)
            if not header:
                break  # socket closed
            body_bytes = client_socket.recv(header.BODY_SIZE)
            data = json.loads(body_bytes)
            if header.SUBJECT_CODE == b"BB_STATE":
                state_updated = True
                ball_balancing_state = BALL_BALANCING_STATE(**data)
            elif header.SUBJECT_CODE == b"BB_REWRD":
                reward_updated = True
                ball_balancing_reward = BALL_BALANCING_REWARD(**data).Reward
            elif header.SUBJECT_CODE == b"BB_DONE_":
                done_updated = True
                ball_balancing_done = BALL_BALANCING_DONE(**data).Done
                # print(f"ball_balancing_done:{ball_balancing_done}")
            # print(f"packet receive:{header.SUBJECT_CODE}")
            if state_updated and reward_updated and done_updated:
                with state_lock:
                    # print(f"{ball_balancing_state.BallPositionX}")
                    ball_balancing_all_updated = True

                state_updated = False
                reward_updated = False
                done_updated = False

    finally:
        client_socket.close()


host = 'localhost'
port = 11200

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((host, port))
print(f"Server start, host:{host}, port:{port}")

server_socket.listen()
current_client = None
current_client, address = server_socket.accept()
print(f"Accepted connection from {address}")

# 각 클라이언트 연결을 처리하기 위한 새 스레드 시작
client_thread = threading.Thread(target=handle_client_connection, args=(current_client,))
client_thread.start()