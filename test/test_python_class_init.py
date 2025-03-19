

class ObservationConfig:
    def __init__(self):
        self.actions = AttrDict()  # 模拟配置对象
        self.velocity_commands = AttrDict()

class AttrDict(dict):
    """支持点号访问的字典"""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

# 初始化配置
cfg = ObservationConfig()

# 模拟动作存储
class RobotController:
    def __init__(self):
        self._raw_actions = [1.0, 0.0]
        self.low_level_actions = [0.0, 0.0]
    
    def get_last_action(self):
        return self.low_level_actions

controller = RobotController()

# 配置观测项 -------------------------------------------------
# 动作观测（动态获取）
cfg.actions.func = lambda dummy_env: controller.get_last_action()
cfg.actions.params = {}

# 速度指令观测（直接读取）
cfg.velocity_commands.func = lambda dummy_env: controller._raw_actions
cfg.velocity_commands.params = {}

# 使用示例 ---------------------------------------------------
# 模拟环境对象（实际不需要使用）
dummy_env = None

print("初始状态：")
print("Action观测:", cfg.actions.func(dummy_env))  # 输出: [0.0, 0.0]
print("Velocity观测:", cfg.velocity_commands.func(dummy_env))  # 输出: [0.0, 0.0]

# 修改动作值
controller.low_level_actions = [1.0, -0.5]
controller._raw_actions = [2.0, 1.5]

print("\n更新后状态：")
print("Action观测:", cfg.actions.func(dummy_env))  # 输出: [1.0, -0.5]
print("Velocity观测:", cfg.velocity_commands.func(dummy_env))  # 输出: [2.0, 1.5]

