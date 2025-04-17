import numpy as np
from rknn.api import RKNN

if __name__ == '__main__':
    # Initialize RKNN object
    rknn = RKNN()

    # Configure the RKNN model
    rknn.config(mean_values=[[0] * (45 + 8 + 4 + 4), [0] * (45 + 8 + 4 + 4) * 10], std_values=[[1] * (45 + 8 + 4 + 4), [1] * (45 + 8 + 4 + 4)*10], target_platform='rk3588')  # Replace with your target platform

    # Load ONNX model
    onnx_model_path = '/home/rl/chenjiawei/wheel/quad_rl/legged_gym/logs/q10w_loco/exported/policies/actor_critic.onnx'  # Replace with your actual ONNX model path
    ret = rknn.load_onnx(model=onnx_model_path)
    if ret != 0:
        print('Load ONNX model failed.')
        exit(ret)


    # Build the RKNN model
    ret = rknn.build(do_quantization=False)  # Set to True if you need quantization
    if ret != 0:
        print('Build RKNN model failed.')
        exit(ret)

    rknn_model_path = '/home/rl/chenjiawei/wheel/quad_rl/legged_gym/logs/q10w_loco/exported/policies/actor_critic.rknn'  # Replace with your desired exported RKNN model path
    # Export the RKNN model
    ret = rknn.export_rknn(rknn_model_path)
    if ret != 0:
        print('Export RKNN model failed.')
        exit(ret)

    print("RKNN model export completed successfully!")
