train=1000
eval_goal=100
eval_traj=20

python bc_goal.py --env MassPointGoal-v1 --name MassPointGoal-v1 --id 00 --num_train $train --num_eval $eval_goal &
python bc_goal.py --env MassPointGoal-v1 --name MassPointGoal-v1 --id 01 --num_train $train --num_eval $eval_goal &
python bc_goal.py --env MassPointGoal-v1 --name MassPointGoal-v1 --id 02 --num_train $train --num_eval $eval_goal &
python bc_goal.py --env MassPointGoal-v1 --name MassPointGoal-v1 --id 03 --num_train $train --num_eval $eval_goal &
python bc_goal.py --env MassPointGoal-v1 --name MassPointGoal-v1 --id 04 --num_train $train --num_eval $eval_goal &
wait

python bc_traj.py --env MassPointTraj-v1 --name MassPointTraj-v1 --id 00 --num_train $train --num_eval $eval_traj &
python bc_traj.py --env MassPointTraj-v1 --name MassPointTraj-v1 --id 01 --num_train $train --num_eval $eval_traj &
python bc_traj.py --env MassPointTraj-v1 --name MassPointTraj-v1 --id 02 --num_train $train --num_eval $eval_traj &
python bc_traj.py --env MassPointTraj-v1 --name MassPointTraj-v1 --id 03 --num_train $train --num_eval $eval_traj &
python bc_traj.py --env MassPointTraj-v1 --name MassPointTraj-v1 --id 04 --num_train $train --num_eval $eval_traj &
wait

python bc_goal.py --env ReacherGoal-v0 --name ReacherGoal-v0 --id 00 --num_train $train --num_eval $eval_goal &
python bc_goal.py --env ReacherGoal-v0 --name ReacherGoal-v0 --id 01 --num_train $train --num_eval $eval_goal &
python bc_goal.py --env ReacherGoal-v0 --name ReacherGoal-v0 --id 02 --num_train $train --num_eval $eval_goal &
python bc_goal.py --env ReacherGoal-v0 --name ReacherGoal-v0 --id 03 --num_train $train --num_eval $eval_goal &
python bc_goal.py --env ReacherGoal-v0 --name ReacherGoal-v0 --id 04 --num_train $train --num_eval $eval_goal &
wait

python bc_goal.py --env ReacherTraj-v0 --name ReacherTraj-v0 --id 00 --num_train $train --num_eval $eval_traj &
python bc_goal.py --env ReacherTraj-v0 --name ReacherTraj-v0 --id 01 --num_train $train --num_eval $eval_traj &
python bc_goal.py --env ReacherTraj-v0 --name ReacherTraj-v0 --id 02 --num_train $train --num_eval $eval_traj &
python bc_goal.py --env ReacherTraj-v0 --name ReacherTraj-v0 --id 03 --num_train $train --num_eval $eval_traj &
python bc_goal.py --env ReacherTraj-v1 --name ReacherTraj-v0 --id 04 --num_train $train --num_eval $eval_traj &
wait

