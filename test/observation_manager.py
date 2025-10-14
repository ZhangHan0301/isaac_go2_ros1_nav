from dataclasses import dataclass
from typing import Callable

# 模拟 ObservationTermCfg（配置类）
@dataclass
class ObservationTermCfg:
    name: str
    func: object   # 可以是函数，也可以是类实例


# 模拟一个函数型观测
def robot_pos():
    return [1.0, 2.0, 3.0]


# 模拟一个类型观测
class ImuSensor:
    def __init__(self):
        self.state = [0.0, 0.0, 9.8]

    def compute(self):
        return self.state

    def reset(self):
        print(f"[ImuSensor] reset called!")
        self.state = [0.0, 0.0, 9.8]


# =============== 模拟管理器 ===============
class ObservationManager:
    def __init__(self):
        # 1. 每个 group 内有哪些 term 名字
        self._group_obs_term_names: dict[str, list[str]] = dict()
        # 2. 每个 term 的维度（shape）
        self._group_obs_term_dim: dict[str, list[tuple[int, ...]]] = dict()
        # 3. 每个 group 内的 term 配置
        self._group_obs_term_cfgs: dict[str, list[ObservationTermCfg]] = dict()
        # 4. 专门存放「类型 term」的配置
        self._group_obs_class_term_cfgs: dict[str, list[ObservationTermCfg]] = dict()
        # 5. 是否需要拼接
        self._group_obs_concatenate: dict[str, bool] = dict()

    def add_term(self, group: str, term_cfg: ObservationTermCfg):
        # 初始化 group
        if group not in self._group_obs_term_names:
            self._group_obs_term_names[group] = []
            self._group_obs_term_cfgs[group] = []
            self._group_obs_class_term_cfgs[group] = []
            self._group_obs_term_dim[group] = []
            self._group_obs_concatenate[group] = True

        # 添加名字
        self._group_obs_term_names[group].append(term_cfg.name)
        self._group_obs_term_cfgs[group].append(term_cfg)

        # 判断是函数还是类
        if callable(term_cfg.func):
            # 函数型：直接获取维度
            obs = term_cfg.func()
            self._group_obs_term_dim[group].append((len(obs),))
        else:
            # 类型：加入 class_term_cfgs，并调用 reset
            self._group_obs_class_term_cfgs[group].append(term_cfg)
            obs = term_cfg.func.compute()
            self._group_obs_term_dim[group].append((len(obs),))
            term_cfg.func.reset()  # ⚡只对类调用 reset

    def summary(self):
        print("=== Term Names ===", self._group_obs_term_names)
        print("=== Term Dims ===", self._group_obs_term_dim)
        print("=== Term Cfgs ===", self._group_obs_term_cfgs)
        print("=== Class Term Cfgs ===", self._group_obs_class_term_cfgs)
        print("=== Concatenate ===", self._group_obs_concatenate)


# =============== 测试 ===============
if __name__ == "__main__":
    manager = ObservationManager()

    # 加一个函数型 term
    manager.add_term("policy", ObservationTermCfg("robot_pos", robot_pos))

    # 加一个类型 term
    manager.add_term("policy", ObservationTermCfg("imu_sensor", ImuSensor()))

    # 加到另一个 group
    manager.add_term("logging", ObservationTermCfg("robot_pos", robot_pos))

    manager.summary()
