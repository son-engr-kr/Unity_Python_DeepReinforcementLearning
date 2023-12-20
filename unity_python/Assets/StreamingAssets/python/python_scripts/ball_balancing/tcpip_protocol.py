import socket
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


