from dataclasses import dataclass
from typing import Type

# ---------- 基础类定义 ----------
class ActionTerm:
    """动作项基类"""
    def __init__(self, cfg, env):
        self.cfg = cfg
        self.env = env

@dataclass
class ActionTermCfg:
    """动作项配置基类"""
    class_type: Type[ActionTerm]

# ---------- 具体实现 ----------
class PreTrainedPolicyAction(ActionTerm):
    """预训练策略动作项实现"""
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        print(f"创建预训练策略动作项: {cfg.asset_name}")
        print(f"策略路径: {cfg.policy_path}")
        print(f"低层级动作配置: {cfg.low_level_actions}")

@dataclass
class PreTrainedPolicyActionCfg(ActionTermCfg):
    """预训练策略配置项"""
    class_type: Type[ActionTerm] = PreTrainedPolicyAction  # 指定实现类
    asset_name: str = "robot_arm"       # 资产名称
    policy_path: str = "policy.pt"      # 策略路径
    low_level_actions: ActionTermCfg = None  # 低层级动作配置

# ---------- 使用示例 ----------
class MockEnv:
    """模拟环境对象"""
    def __init__(self):
        self.scene = {"robot_arm": "articulation_object"}
        self.device = "cuda:0"

if __name__ == "__main__":
    # 创建配置实例
    cfg = PreTrainedPolicyActionCfg(
        low_level_actions=ActionTermCfg(class_type=ActionTerm)
    )

    # 实例化动作项
    env = MockEnv()
    action_term = cfg.class_type(cfg, env)

    # 模拟使用
    print("\n运行结果：")
    print(f"动作项类型: {type(action_term).__name__}")
    print(f"配置验证: {action_term.cfg.asset_name == 'robot_arm'}")