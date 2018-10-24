train=100
eval_goal=100
eval_traj=20
human_demo_size=$1

# MassPointGoal-v1
env_id="MassPointGoal-v1"
expert_name="MassPointGoal-v1"
save_name="MassPointGoal-v1-plot3"
type="goal"
for id in 00 01 02 03 04; do
	python bc_merge_3.py --env-id $env_id --expert-name $expert_name --save-name $save_name --random-seed $id --num-train $train --num-eval $eval_goal --type $type --human_demo_size $human_demo_size &
done 
wait %1 %2 %3 %4 %5

# MassPointGoalInstr-v1
env_id="MassPointGoalInstr-v1"
expert_name="MassPointGoalInstr-v1"
save_name="MassPointGoalInstr-v1-plot3"
type="goal"
model_path="./Asset/model/MassPointGoal-v1-plot3"
for id in 00 01 02 03 04; do
	python bc_merge_3.py --env-id $env_id --expert-name $expert_name --save-name $save_name --random-seed $id --num-train $train --num-eval $eval_goal --type $type --model-path $model_path --human_demo_size $human_demo_size &
done 
wait %1 %2 %3 %4 %5

# MassPointGoalAction-v1
env_id="MassPointGoalAction-v1"
expert_name="MassPointGoalAction-v1"
save_name="MassPointGoalAction-v1-plot3"
type="goal"
model_path="./Asset/model/MassPointGoal-v1-plot3"
for id in 00 01 02 03 04; do
	python bc_merge_3.py --env-id $env_id --expert-name $expert_name --save-name $save_name --random-seed $id --num-train $train --num-eval $eval_goal --type $type --model-path $model_path --human_demo_size $human_demo_size &
done 
wait %1 %2 %3 %4 %5

# MassPointTraj-v1
env_id="MassPointTraj-v1"
expert_name="MassPointTraj-v1"
save_name="MassPointTraj-v1-plot3"
type="traj"
for id in 00 01 02 03 04; do
	python bc_merge_3.py --env-id $env_id --expert-name $expert_name --save-name $save_name --random-seed $id --num-train $train --num-eval $eval_goal --type $type --human_demo_size $human_demo_size &
done
wait %1 %2 %3 %4 %5

# MassPointTrajInstr-v1
env_id="MassPointTrajInstr-v1"
expert_name="MassPointTrajInstr-v1"
save_name="MassPointTrajInstr-v1-plot3"
type="traj"
model_path="./Asset/model/MassPointTraj-v1-plot3"
for id in 00 01 02 03 04; do
	python bc_merge_3.py --env-id $env_id --expert-name $expert_name --save-name $save_name --random-seed $id --num-train $train --num-eval $eval_goal --type $type --model-path $model_path --human_demo_size $human_demo_size &
done 
wait %1 %2 %3 %4 %5

# MassPointTrajAction-v1
env_id="MassPointTrajAction-v1"
expert_name="MassPointTrajAction-v1"
save_name="MassPointTrajAction-v1-plot3"
type="traj"
model_path="./Asset/model/MassPointTraj-v1-plot3"
for id in 00 01 02 03 04; do
	python bc_merge_3.py --env-id $env_id --expert-name $expert_name --save-name $save_name --random-seed $id --num-train $train --num-eval $eval_goal --type $type --model-path $model_path --human_demo_size $human_demo_size &
done 
wait %1 %2 %3 %4 %5
