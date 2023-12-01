print("hello", flush=True)
print("hello2\nhello3\nhello4", flush=True)

import numpy as np
# import torch
import time
import sys
idx = 0
print(f"sys.path: {sys.path}", flush=True)
# print(f"torch cuda: {torch.cuda.is_available()}({torch.version.cuda}), torch version: {torch.__version__}")
print(np.array([1,2,3]))

for idx in range(10):
    print(f"Hello this is python process from unity: {idx}", flush=True)
    idx += 1
    time.sleep(1)
for idx in range(3):
    print("/input", flush=True)
    input_string = input()
    print(f"input result: {input_string}", flush=True)
    time.sleep(3)
