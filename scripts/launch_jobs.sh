#!/bin/bash
sweep="wandb agent inouye-lab/AISTATS24-IL-ndevice/qnxu3okn"
task_dir="/local/scratch/a/zhou1059/InternetLearning"


for machine in eggplant cranberry dewberry avocado fig
do
    for device in 0 1
    do
        for repeat in 0
        do
            echo "run task on ${machine} d${device}-${repeat} ${sweep}"
            ssh $machine "tmux send-keys -t "d${device}-${repeat}" 'conda deactivate' ENTER"
            ssh $machine "tmux send-keys -t "d${device}-${repeat}" 'conda activate internet' ENTER"
            ssh $machine "tmux send-keys -t "d${device}-${repeat}" 'cd ${task_dir}' ENTER"
            ssh $machine "tmux send-keys -t "d${device}-${repeat}" 'CUDA_VISIBLE_DEVICES=${device} ${sweep}' ENTER"
        done
    done
done

for machine in grapefruit
do
    for device in 0
    do
        for repeat in 0
        do
            echo "run task on ${machine} d${device}-${repeat} ${sweep}"
            ssh $machine "tmux send-keys -t "d${device}-${repeat}" 'conda deactivate' ENTER"
            ssh $machine "tmux send-keys -t "d${device}-${repeat}" 'conda activate internet' ENTER"
            ssh $machine "tmux send-keys -t "d${device}-${repeat}" 'cd ${task_dir}' ENTER"
            ssh $machine "tmux send-keys -t "d${device}-${repeat}" 'CUDA_VISIBLE_DEVICES=${device} ${sweep}' ENTER"
        done
    done
done

for machine in ichigo honeydew
do
    for device in 0 1 2 3
    do
        for repeat in 0
        do
            echo "run task on ${machine} d${device}-${repeat} ${sweep}"
            ssh $machine "tmux send-keys -t "d${device}-${repeat}" 'conda deactivate' ENTER"
            ssh $machine "tmux send-keys -t "d${device}-${repeat}" 'conda activate internet' ENTER"
            ssh $machine "tmux send-keys -t "d${device}-${repeat}" 'cd ${task_dir}' ENTER"
            ssh $machine "tmux send-keys -t "d${device}-${repeat}" 'CUDA_VISIBLE_DEVICES=${device} ${sweep}' ENTER"
        done
    done
done
