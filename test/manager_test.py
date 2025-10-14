from dataclasses import dataclass

# 模拟 mdp 配置
@dataclass
class PreTrainedPolicyActionCfg:
    asset_name: str
    low_level_observations: str


# 模拟 @configclass 装饰器（其实就是 dataclass）
def configclass(cls):
    return dataclass(cls)


@configclass
class ActionsCfg:
    """Action terms for the MDP."""
    pre_trained_policy_action: PreTrainedPolicyActionCfg = PreTrainedPolicyActionCfg(
        asset_name="robot",
        low_level_observations="policy_obs",
    )


# ========================
# TermManager 和 ActionTerm
class ActionTerm:
    def __init__(self, cfg):
        self.cfg = cfg
    def __repr__(self):
        return f"ActionTerm(cfg={self.cfg})"


class TermManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self._term_names = []
        self._terms = {}

    def _prepare_terms(self):
        if isinstance(self.cfg, dict):
            cfg_items = self.cfg.items()
        else:  # dataclass 对象
            cfg_items = self.cfg.__dict__.items()
            print(cfg_items)

        for name, term_cfg in cfg_items:
            print("name: ", name)
            print(term_cfg)
            self._term_names.append(name)
            self._terms[name] = ActionTerm(term_cfg)

    def show(self):
        print("Term names:", self._term_names)
        print("Terms dict:", self._terms)


# ========================
# 测试
actions_cfg = ActionsCfg()
manager = TermManager(actions_cfg)
manager._prepare_terms()
manager.show()
