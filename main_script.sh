#!/bin/bash

if [ "$1" = "start" ]; then
    for i in {0..19..1}; do
        echo "dramatiq -p 1 models.m$i --log-file logs/m$i.log &"
        nohup dramatiq -p 1 models.m$i --log-file logs/m$i.log &
    done
elif [ "$1" = "stop" ]; then
    ps aux | grep drama | awk '{print $2}' | xargs kill
elif [ "$1" = "init" ]; then
    mkdir -p models
    mkdir -p logs
    cp inpainting.py models/
    for i in {0..19..1}; do
        cat main_script_prefix.py > models/m$i.py
        cat >> models/m$i.py << EOF
import dramatiq
@dramatiq.actor(queue_name='m<cat_id>', store_results=True, max_retries=0)
def predict(raw_image):
    from inpainting import Inpainting
    model = Inpainting.from_cat_id(<cat_id>)
    return model.predict(raw_image)
EOF
        sed -i "s/<cat_id>/$i/g" "models/m$i.py"
    done
fi
