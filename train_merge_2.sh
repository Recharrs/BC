train=1000
eval_goal=100
eval_traj=20

env="MassPointGoal-v1"
name="MassPointGoal-v1"
save_name="MassPointGoal-v1-3"
type="goal"
for id in 00 01 02 03 04; do
  python bc_merge_2.py --env $env --name $name --save-name $save_name --id $id --num_train $train --num_eval $eval_goal --type $type &
done 
wait

env="MassPointTraj-v1"
name="MassPointTraj-v1"
save_name="MassPointTraj-v1-3"
type="traj"
for id in 00 01 02 03 04; do
  python bc_merge_2.py --env $env --name $name --save-name $save_name --id $id --num_train $train --num_eval $eval_goal --type $type &
done
wait

# env="ReacherGoal-v1"
# name="ReacherGoal-v1"
# save_name="ReacherGoal-v1-3"
# type="goal"
# for id in 00 01 02 03 04; do
#   python bc_merge_2.py --env $env --name $name --save-name $save_name --id $id --num_train $train --num_eval $eval_goal --type $type &
# done 
# wait

# env="ReacherTraj-v1"
# name="ReacherTraj-v1"
# save_name="ReacherTraj-v1-3"
# type="traj"
# for id in 00 01 02 03 04; do
#   python bc_merge_2.py --env $env --name $name --save-name $save_name --id $id --num_train $train --num_eval $eval_goal --type $type &
# done
# wait
