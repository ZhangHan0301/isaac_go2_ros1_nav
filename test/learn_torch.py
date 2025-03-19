import torch

# 修复后的代码
loaded_model = torch.jit.load("ckpts/unitree_go2/rough_model_7850.pt")  # ✅ 先加载
torch.jit.save(loaded_model, "ckpts/unitree_go2/rough_go2_jit.pt")      # ✅ 再保存