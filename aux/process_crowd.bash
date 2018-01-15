#!/bin/bash
# https://unix.stackexchange.com/questions/146833/setting-running-process-affinity-with-taskset-fails
sudo ps -eLo cmd,tid | perl -pe 's/.* (\d+)$/\1/' | xargs -n 1 sudo taskset -cp 0
