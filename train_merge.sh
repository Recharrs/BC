train=1000
eval_goal=100
eval_traj=20

env="MassPointGoal-v1"
name="MassPointGoal-v1"
for id in 00 01 02 03 04; do
  python bc_goal.py --env $env --name $name --id $id --num_train $train --num_eval $eval_goal &
done 
wait

env="MassPointTraj-v1"
name="MassPointTraj-v1"
for id in 00 01 02 03 04; do
  python bc_traj.py --env $env --name $name --id $id --num_train $train --num_eval $eval_goal &
done
wait

env="ReacherGoal-v1"
name="ReacherGoal-v1"
for id in 00 01 02 03 04; do
  python bc_goal.py --env $env --name $name --id $id --num_train $train --num_eval $eval_goal &
done 
wait

env="ReacherTraj-v1"
name="ReacherTraj-v1"
for id in 00 01 02 03 04; do
  python bc_traj.py --env $env --name $name --id $id --num_train $train --num_eval $eval_goal &
done
wait
