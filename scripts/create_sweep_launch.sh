#!/bin/bash

task_dir="/local/scratch/a/zhou1059/ild_domaincf/images"
#sweep_dir="./configs/app_testfault_ndevice.yaml"
read -p "Enter the sweep directory path: " input_sweep_dir
sweep_dir="$task_dir/configs/$input_sweep_dir"
#sweep_dir=$(echo "$input_sweep_dir" | sed 's/ /\\ /g')

sweep_output=$(wandb sweep $sweep_dir 2>&1)
# Extract the sweep ID from the captured output
sweep_id=$(echo "$sweep_output" | grep -o 'wandb: Created sweep with ID: [^ ]*' | awk '{print $6}')
sweep_url=$(echo "$sweep_output" | grep -o 'wandb: View sweep at: [^ ]*' | awk '{print $5}')
sweep_agent_command=$(echo "$sweep_output" | grep -o 'wandb: Run sweep agent with: .*' | sed 's/wandb: Run sweep agent with: //')

# Print the captured sweep ID, URL, and sweep agent command
echo "Sweep ID: $sweep_id"
echo "Sweep URL: $sweep_url"
echo "Sweep Agent Command: $sweep_agent_command"


for machine in eggplant fig  # cranberry dewberry avocado blueberry
do
    for device in 0 1
    do
        for repeat in 0
        do
            echo "run task on ${machine} d${device}-${repeat} ${sweep_agent_command}"
            ssh $machine "tmux send-keys -t "d${device}-${repeat}" 'conda deactivate' ENTER"
            ssh $machine "tmux send-keys -t "d${device}-${repeat}" 'conda activate causal' ENTER"
            ssh $machine "tmux send-keys -t "d${device}-${repeat}" 'cd ${task_dir}' ENTER"
            ssh $machine "tmux send-keys -t "d${device}-${repeat}" 'git pull' ENTER"
            ssh $machine "tmux send-keys -t "d${device}-${repeat}" 'CUDA_VISIBLE_DEVICES=${device} ${sweep_agent_command}' ENTER"
        done
    done
done

for machine in grapefruit
do
    for device in 0
    do
        for repeat in 0
        do
            echo "run task on ${machine} d${device}-${repeat} ${sweep_agent_command}"
            ssh $machine "tmux send-keys -t "d${device}-${repeat}" 'conda deactivate' ENTER"
            ssh $machine "tmux send-keys -t "d${device}-${repeat}" 'conda activate causal' ENTER"
            ssh $machine "tmux send-keys -t "d${device}-${repeat}" 'cd ${task_dir}' ENTER"
            ssh $machine "tmux send-keys -t "d${device}-${repeat}" 'git pull' ENTER"
            ssh $machine "tmux send-keys -t "d${device}-${repeat}" 'CUDA_VISIBLE_DEVICES=${device} ${sweep_agent_command}' ENTER"
        done
    done
done

for machine in honeydew ichigo
do
    for device in 0 1 2 3
    do
        for repeat in 0
        do
            echo "run task on ${machine} d${device}-${repeat} ${sweep_agent_command}"
            ssh $machine "tmux send-keys -t "d${device}-${repeat}" 'conda deactivate' ENTER"
            ssh $machine "tmux send-keys -t "d${device}-${repeat}" 'conda activate causal' ENTER"
            ssh $machine "tmux send-keys -t "d${device}-${repeat}" 'cd ${task_dir}' ENTER"
            ssh $machine "tmux send-keys -t "d${device}-${repeat}" 'git pull' ENTER"
            ssh $machine "tmux send-keys -t "d${device}-${repeat}" 'CUDA_VISIBLE_DEVICES=${device} ${sweep_agent_command}' ENTER"
        done
    done
done

