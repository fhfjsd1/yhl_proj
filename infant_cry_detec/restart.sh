#!/bin/bash

while true; do
    PID=$(ps -aux | grep '/home/taylor/infant_speechbrain/SoundClassification/train.py' | grep -v 'grep' | awk '{print $2}')
    if [ -n "$PID" ]; then
        echo "running right"
        sleep 180
else
    echo "Restarting Python script"
    /home/taylor/anaconda3/bin/python /home/taylor/infant_speechbrain/SoundClassification/train.py /home/taylor/infant_speechbrain/SoundClassification/hparams.yaml &
    sleep 180
fi

done

