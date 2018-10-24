train=1000
eval_goal=100
eval_traj=20
human_demo_size=$1
root_dir=$2

# ReacherGoal-v0
env_id="ReacherGoal-v0"
expert_name="ReacherGoal-v0"
save_name="ReacherGoal-v0-plot3"
type="goal"
for id in 00 01 02 03 04; do
	python bc/bc_merge_3.py --env-id $env_id --expert-name $expert_name --save-name $save_name --random-seed $id --num-train $train --num-eval $eval_goal --type $type --human_demo_size $human_demo_size --root $root_dir &
done 
wait %1 %2 %3 %4 %5 

# ReacherGoalInstr-v0
env_id="ReacherGoalInstr-v0"
expert_name="ReacherGoalInstr-v0"
save_name="ReacherGoalInstr-v0-plot3"
type="goal"
model_path="ReacherGoal-v0-plot3"
for id in 00 01 02 03 04; do
	python bc/bc_merge_3.py --env-id $env_id --expert-name $expert_name --save-name $save_name --random-seed $id --num-train $train --num-eval $eval_goal --type $type --model-path $model_path --human_demo_size $human_demo_size --root $root_dir &
done 
wait %1 %2 %3 %4 %5

# ReacherGoalAction
env_id="ReacherGoalAction-v0"
expert_name="ReacherGoalAction-v0"
save_name="ReacherGoalAction-v0-plot3"
type="goal"
model_path="ReacherGoal-v0-plot3"
for id in 00 01 02 03 04; do
	python bc/bc_merge_3.py --env-id $env_id --expert-name $expert_name --save-name $save_name --random-seed $id --num-train $train --num-eval $eval_goal --type $type --model-path $model_path --human_demo_size $human_demo_size --root $root_dir &
done 
wait %1 %2 %3 %4 %5

# ReacherGoal-v0
env_id="ReacherTraj-v0"
expert_name="ReacherTraj-v0"
save_name="ReacherTraj-v0-plot3"
type="traj"
for id in 00 01 02 03 04; do
	python bc/bc_merge_3.py --env-id $env_id --expert-name $expert_name --save-name $save_name --random-seed $id --num-train $train --num-eval $eval_goal --type $type --human_demo_size $human_demo_size --root $root_dir &
done 
wait %1 %2 %3 %4 %5

# ReacherGoalInstr-v0
env_id="ReacherTrajInstr-v0"
expert_name="ReacherTrajInstr-v0"
save_name="ReacherTrajInstr-v0-plot3"
type="traj"
model_path="ReacherTraj-v0-plot3"
for id in 00 01 02 03 04; do
	python bc/bc_merge_3.py --env-id $env_id --expert-name $expert_name --save-name $save_name --random-seed $id --num-train $train --num-eval $eval_goal --type $type --model-path $model_path --human_demo_size $human_demo_size --root $root_dir &
done 
wait %1 %2 %3 %4 %5

# ReacherTrajAction-v0
env_id="ReacherTrajAction-v0"
expert_name="ReacherTrajAction-v0"
save_name="ReacherTrajAction-v0-plot3"
type="traj"
model_path="ReacherTraj-v0-plot3"
for id in 00 01 02 03 04; do
	python bc/bc_merge_3.py --env-id $env_id --expert-name $expert_name --save-name $save_name --random-seed $id --num-train $train --num-eval $eval_goal --type $type --model-path $model_path --human_demo_size $human_demo_size --root $root_dir &
done 
wait %1 %2 %3 %4 %5
