docker run \
    --rm -it --gpus all \
    --shm-size=4g \
    -u "$(id -u $USER):$(id -g $USER)" \
    -v $(pwd):/work \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/group:/etc/group:ro \
    -v /home/$USER:/home/$USER \
    -w /work/samples \
    imx500-zoo:latest \
    /bin/bash
