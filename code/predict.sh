OUTPUT_DIR="/result"
DATA_DIR="/data"
CODE_DIR="/code"

if [[ $# -eq 3  && $1 == "--local" ]] ; then
    DATA_DIR=$2
    OUTPUT_DIR=$3
    CODE_DIR=$4
fi

echo "Load data from $DATA_DIR"
echo "Output will be saved to $OUTPUT_DIR"

# Create submission directory if not exist
if [ ! -d $OUTPUT_DIR ]; then
    echo "$OUTPUT_DIR not exist, creating..."
    mkdir $OUTPUT_DIR
fi

echo "Run prediction..."
python3 $CODE_DIR/predict.py --song_dir $DATA_DIR/songs --lyric_dir $DATA_DIR/lyrics --output_dir $OUTPUT_DIR

echo "Changing directory to $OUTPUT_DIR"
cd $OUTPUT_DIR

echo "Zipping..."
mkdir submission
mv *.json submission
zip -r submission.zip submission
rm -fr submission

echo "Done"
