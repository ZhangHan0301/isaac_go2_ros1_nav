
```
Isaac-Navigation-Flat-go2-v0
# 测试训练结果
python rsl_rl/play.py --task Isaac-Navigation-Flat-Anymal-C-v0 --num_envs 1 --checkpoint ckpts/unitree_go2/anymal_c_navigation/model_9999.pt
#环境go2 
python rsl_rl/play.py --task Isaac-Navigation-Flat-go2-v0 --num_envs 1 --checkpoint ckpts/unitree_go2/anymal_c_navigation/model_9999.pt
python rsl_rl/play.py --task Isaac-Navigation-Flat-go2-v0 --num_envs 1 --checkpoint rsl_rl/logs/rsl_rl/anymal_c_navigation/2025-03-17_14-34-51/model_9999.pt


```

```
# 训练导航策略
python rsl_rl/train.py --task Isaac-Navigation-Flat-Anymal-C-v0 --num_envs 4090 --headless
```

# 原网络输入
+---------------------------------------------------------+
| Active Observation Terms in Group: 'policy' (shape: (10,)) |
+-----------+---------------------------------+-----------+
|   Index   | Name                            |   Shape   |
+-----------+---------------------------------+-----------+
|     0     | base_lin_vel                    |    (3,)   |
|     1     | projected_gravity               |    (3,)   |
|     2     | pose_command                    |    (4,)   |
+-----------+---------------------------------+-----------+
