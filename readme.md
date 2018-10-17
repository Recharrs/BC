# Behavior Cloning

BC of Rule-Base policy or RL Expert.

## TODO

###  Base

- [ ] MassPointGoal
- [ ] MassPointTraj
- [ ] ReacherGoal
- [ ] ReacherTraj

### Change Instruction

- [ ] MassPointGoalInstr

- [ ] MassPointTrajInstr
- [ ] ReacherGoalInstr
- [ ] ReacherTrajInstr

### Change Action
- [ ] MassPointGoalAction
- [ ] MassPointTrajAction


- [ ] ReacherGoalAction
- [ ] ReacherTrajAction

## Experiment

1. total 2 * 6 = 12 experiments
2. each experiment has 5 different kinds of dataset
3. train BC

## Dataset

Total 2500 trajectories for each dataset, each trajectory has its own length.

### MassPoint

- 00 ~ 04: from rule-based policy

### Reacher

- 00 ~ 04: from RL

## Instructions

```
# collect dataset
export DISPLAY=:1; python mass_point_goal_rule_base.py
# preprocess
python preprocess.py
# train BC
py3env; cd bc
python bc_goal.py --env MassPointGoal-v0 --name MassPointGoal-v0/02 --num_train 1000 --num_eval 100 
python bc_traj.py --env MassPointTraj-v0 --name MassPointTraj-v0 --id 00 --num_train 1 --num_eval 100 
python -m baselines.run --alg=ppo2 --env=ReacherGoal-v0 --num_timesteps=3e6 --nsteps=1000 --save_path=Model/ReacherGoal-v0 --save_interval=1
```
