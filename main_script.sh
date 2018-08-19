#!/bin/bash

if [ "$1" = "start" ]; then
    for i in {0..19..1}; do
        dramatiq -p 1 models.m$i --log-file logs/m$i.log &
    done
elif [ "$1" = "stop" ]; then
    ps aux | grep drama | awk '{print $2}' | xargs kill
elif [ "$1" = "init" ]; then
    for i in {0..19..1}; do
        mkdir models
        mkdir logs
        cp inpainting.py models/
        echo -e "import dramatiq\nfrom inpainting import Inpainting\n@dramatiq.actor(queue_name='m$i')\ndef predict(raw_image):\n\tmodel = Inpanting.from_cat_id($i)\n\tresult_img = model.predict(raw_image)" > models/m$i.py
    done
fi
