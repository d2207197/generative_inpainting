#!/bin/bash

if [ "$1" = "start" ]; then
    for i in {0..19..1}; do
        echo "dramatiq -p 1 models.m$i --log-file logs/m$i.log &"
        nohup dramatiq -p 1 models.m$i --log-file logs/m$i.log &
    done
elif [ "$1" = "stop" ]; then
    ps aux | grep drama | awk '{print $2}' | xargs kill
elif [ "$1" = "init" ]; then
    mkdir models
    mkdir logs
    cp inpainting.py models/
    for i in {0..19..1}; do
        cat main_script_prefix.py > models/m$i.py
        echo -e "\nfrom inpainting import Inpainting\n@dramatiq.actor(queue_name='m$i', store_results=True)\ndef predict(raw_image):\n\traw_image = cv2.imread(raw_image)\n\tmodel = Inpanting.from_cat_id($i)\n\treturn model.predict(raw_image)" >> models/m$i.py
    done
fi
