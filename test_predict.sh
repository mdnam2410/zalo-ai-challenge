if [[ $# -eq 3 && $1 == "--local" ]] ; then
    ./code/predict.sh --local $2 $3
else
    sudo docker run --gpus "device=0" -v $1:/data -v $2:/result zac2022 /bin/bash /code/predict.sh
fi
