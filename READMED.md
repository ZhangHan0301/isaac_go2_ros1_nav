
```
Isaac-Navigation-Flat-go2-v0
# 测试训练结果
python rsl_rl/play.py --task Isaac-Navigation-Flat-Anymal-C-v1 --num_envs 1 --checkpoint ckpts/unitree_go2/anymal_c_navigation/model_9999.pt

#环境go2 
python rsl_rl/play.py --task Isaac-Navigation-go2-v0 --num_envs 1 --checkpoint ckpts/unitree_go2/anymal_c_navigation/model_9999.pt
python rsl_rl/play.py --task Isaac-Navigation-go2-play-v0 --num_envs 1 --checkpoint rsl_rl/logs/rsl_rl/anymal_c_navigation/2025-03-17_14-34-51/model_9999.pt


```

```
# 训练导航策略
python rsl_rl/train.py --task Isaac-Navigation-go2-v0 --num_envs 4090 --headless
```
#断点重新训练
python rsl_rl/train.py --task Isaac-Navigation-go2-v0 --num_envs 3000 --headless --resume true --checkpoint logs/rsl_rl/unitree_go2_rough/2025-03-21_14-48-56/model_2000.pt
python rsl_rl/train.py --task Isaac-Navigation-Flat-Anymal-C-v1 --num_envs 4090 --headless

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

# 查看日志
python -m tensorboard.main --logdir logs/rsl_rl/unitree/2025-03-20_17-31-12/