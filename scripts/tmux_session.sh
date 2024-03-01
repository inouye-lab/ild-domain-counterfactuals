#!/bin/bash

for machine in cranberry dewberry eggplant fig #  avocado blueberry
do
    for device in 0 1
    do
        for repeat in 0 1
        do
            echo "generate tmux session ${machine} d${device}-${repeat}"
            ssh $machine "tmux new-session -d -s d${device}-${repeat}"
        done
    done
done

for machine in grapefruit
do
    for device in 0
    do
        for repeat in 0 1
        do
            echo "generate tmux session ${machine} d${device}-${repeat}"
            ssh $machine "tmux new-session -d -s d${device}-${repeat}"
        done
    done
done

for machine in honeydew ichigo
do
    for device in 0 1 2 3
    do
        for repeat in 0 1
        do
            echo "generate tmux session ${machine} d${device}-${repeat}"
            ssh $machine "tmux new-session -d -s d${device}-${repeat}"
        done
    done
done