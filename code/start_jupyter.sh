CODE_DIR='/code'
echo "cd into $CODE_DIR and activate virtual environment"
cd $CODE_DIR
source bin/activate

echo "Return to root"
cd /

jupyter lab --port 9777 --ip 0.0.0.0 --NotebookApp.password='zac2022' --NotebookApp.token='zac2022' --allow-root --no-browser
