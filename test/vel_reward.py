import numpy as np

def compute_velocity_reward(P_g, P_r, V_r, lambda1=1.0, lambda2=1.0):
    """
    修改后的奖励函数（二维版）
    :param P_g: 目标位置，np.array([x, y])
    :param P_r: 当前位置，np.array([x, y])
    :param V_r: 当前速度，np.array([v_x, v_y])
    :param lambda1: X方向超速惩罚系数
    :param lambda2: Y方向移动惩罚系数
    :return: 奖励值
    """
    delta = P_g - P_r
    distance = np.linalg.norm(delta)
    
    # 处理目标与当前位置重合的情况
    if distance < 1e-6:
        return 0.0
    
    # 方向对齐奖励
    unit_direction = delta / distance
    alignment_reward = np.dot(unit_direction, V_r)
    
    # X方向超速惩罚：v_x > 1.0时施加二次惩罚
    v_x = V_r[0]
    speed_penalty = lambda1 * max(v_x - 1.0, 0) ** 2
    
    # Y方向移动惩罚：对任何v_y施加二次惩罚
    v_y = V_r[1]
    lateral_penalty = lambda2 * v_y ** 2
    
    # 总奖励 = 对齐奖励 - 惩罚项
    total_reward = alignment_reward - speed_penalty - lateral_penalty
    return total_reward

# --------------------------
# 测试用例
# --------------------------
def test_case(case_name, P_g, P_r, V_r, expected):
    reward = compute_velocity_reward(P_g, P_r, V_r)
    print(f"[{case_name}]")
    print(f"目标位置 P_g: {P_g}")
    print(f"当前位置 P_r: {P_r}")
    print(f"当前速度 V_r: {V_r}")
    print(f"计算奖励: {reward:.2f} | 预期奖励: {expected}")
    print("-" * 50)

if __name__ == "__main__":
    # 用例1：X方向对齐且未超速（最佳情况）
    test_case(
        case_name="理想正向移动",
        P_g=np.array([5.0, 0.0]),
        P_r=np.array([0.0, 0.0]),
        V_r=np.array([1.0, 0.0]),  # v_x=1.0, v_y=0
        expected=1.0  # 对齐奖励=1.0, 无惩罚
    )

    # 用例2：X方向超速（v_x=1.5 > 1.0）
    test_case(
        case_name="X方向超速",
        P_g=np.array([3.0, 0.0]),
        P_r=np.array([0.0, 0.0]),
        V_r=np.array([1.5, 0.0]),
        expected=1.5 - 1.0*(1.5-1.0)**2  # 1.5 - 0.25 = 1.25
    )

    # 用例3：横向移动（v_y=2.0）
    test_case(
        case_name="横向移动",
        P_g=np.array([2.0, 0.0]),
        P_r=np.array([0.0, 0.0]),
        V_r=np.array([0.0, 2.0]),  # 完全横向
        expected=0.0 - 0 - 1.0*(2.0)**2  # 0 - 0 - 4.0 = -4.0
    )

    # 用例4：混合情况（v_x=0.8未超速，v_y=1.0）
    test_case(
        case_name="混合移动",
        P_g=np.array([2.0, 0.0]),
        P_r=np.array([1.0, 0.0]),
        V_r=np.array([0.8, 1.0]),
        expected=0.8*(1.0/1.0) - 0 - 1.0*(1.0)**2  # 0.8 - 1.0 = -0.2
    )