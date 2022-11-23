sudo docker run -it --gpus "device=0" -p9777:9777 -v $1:/data -v $2:/result zac2022 /bin/bash /code/start_jupyter.sh
