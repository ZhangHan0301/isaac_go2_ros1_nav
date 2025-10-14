from dataclasses import dataclass, field
from typing import Dict, Type, List
import numpy as np

# ---------------------------
# 抽象接口：所有动作模块的统一协议
# ---------------------------
class ActionTerm:
    def __init__(self, cfg, env=None):
        self.cfg = cfg
        self.env = env
        self._raw_actions = None

    @property
    def action_dim(self) -> int:
        """每个term需要的动作维度（由子类实现）"""
        raise NotImplementedError

    def process_actions(self, actions: np.ndarray):
        """接收该term的动作片段（由子类实现）"""
        raise NotImplementedError

    def apply_actions(self):
        """真正执行/打印/下发动作（由子类实现）"""
        raise NotImplementedError


# ---------------------------
# 具体动作模块实现（可互换）
# ---------------------------
class MoveBaseAction(ActionTerm):
    """移动底盘：vx, vy, wz 三个自由度"""
    @property
    def action_dim(self) -> int:
        return 3

    def process_actions(self, actions: np.ndarray):
        # 做一些限制或预处理
        self._raw_actions = np.clip(actions, -1.0, 1.0)

    def apply_actions(self):
        vx, vy, wz = self._raw_actions[0]
        print(f"[MoveBase] apply vx={vx:.2f}, vy={vy:.2f}, wz={wz:.2f}")


class ArmGripperAction(ActionTerm):
    """机械臂末端夹爪：开合1维"""
    @property
    def action_dim(self) -> int:
        return 1

    def process_actions(self, actions: np.ndarray):
        # 将 [-1,1] 映射为 [0,1] 的开合百分比
        x = np.clip(actions[0, 0], -1.0, 1.0)
        self._raw_actions = np.array([[0.5 * (x + 1.0)]])

    def apply_actions(self):
        grip = self._raw_actions[0, 0]
        print(f"[Gripper] apply open_ratio={grip:.2f}")


# ---------------------------
# “配置类” —— 关键是 class_type 指向要创建的类
# ---------------------------
@dataclass
class ActionTermCfg:
    class_type: Type[ActionTerm]                 # 指定要实例化哪个动作类
    name: str                                    # 该term的名字
    params: dict = field(default_factory=dict)   # 其它参数（可选）


# ---------------------------
# 管理器：根据配置“工厂化”创建 term，并分发动作
# ---------------------------
class ActionManager:
    def __init__(self, cfgs: Dict[str, ActionTermCfg]):
        self._term_names: List[str] = []
        self._terms: Dict[str, ActionTerm] = {}
        self._prepare_terms(cfgs)

    def _prepare_terms(self, cfgs: Dict[str, ActionTermCfg]):
        for term_name, term_cfg in cfgs.items():
            if term_cfg is None:
                continue
            if not issubclass(term_cfg.class_type, ActionTerm):
                raise TypeError(f"{term_name}: class_type must be ActionTerm subclass.")
            term = term_cfg.class_type(term_cfg, env=None)  # 传env也行，这里简化
            self._term_names.append(term_name)
            self._terms[term_name] = term

        # 计算总动作维度
        self.total_action_dim = sum(self._terms[n].action_dim for n in self._term_names)

    def step(self, action: np.ndarray):
        """将总动作向量切片，分发给每个term并执行"""
        assert action.shape == (1, self.total_action_dim), \
            f"action shape expected (1,{self.total_action_dim}), got {action.shape}"

        idx = 0
        for name in self._term_names:
            term = self._terms[name]
            sub = action[:, idx: idx + term.action_dim]
            term.process_actions(sub)
            idx += term.action_dim

        # 统一调用 apply
        for name in self._term_names:
            self._terms[name].apply_actions()

    def describe(self):
        print("=== ActionManager Layout ===")
        cum = 0
        for name in self._term_names:
            dim = self._terms[name].action_dim
            print(f"{name:>12s}: dim={dim}, slice=[{cum}:{cum+dim}]")
            cum += dim
        print(f"Total action dim: {self.total_action_dim}\n")


# ---------------------------
# 演示：仅改配置即可替换控制模块
# ---------------------------
def demo():
    # 1) 配置：底盘(3维) + 夹爪(1维)
    cfgs = {
        "base": ActionTermCfg(class_type=MoveBaseAction, name="base"),
        "grip": ActionTermCfg(class_type=ArmGripperAction, name="grip"),
    }
    manager = ActionManager(cfgs)
    manager.describe()

    # 2) 准备一个总动作向量（维度=3+1=4）
    #    [vx, vy, wz, grip_cmd]
    action = np.array([[0.8, -0.2, 1.3, -0.5]], dtype=float)
    print("Input action:", action)

    # 3) 一步执行：自动切片并调用各term
    manager.step(action)

    # 4) 替换策略：把底盘换成“ArmGripperAction”只是为了演示可替换性（不合理，但能看模式）
    print("\n--- Swap base term class via config ---")
    cfgs2 = {
        "base": ActionTermCfg(class_type=ArmGripperAction, name="base_as_gripper"),
        "grip": ActionTermCfg(class_type=ArmGripperAction, name="grip"),
    }
    manager2 = ActionManager(cfgs2)
    manager2.describe()
    action2 = np.array([[0.1, 0.9]], dtype=float)  # 1+1 = 2 维
    manager2.step(action2)


if __name__ == "__main__":
    demo()
