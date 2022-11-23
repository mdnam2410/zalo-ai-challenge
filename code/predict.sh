OUTPUT_DIR="./result"
DATA_DIR="./data"

echo "Loading data from $DATA_DIR"

# Create submission directory if not exist
if [ ! -d $SUBMISSION_DIR ]; then
    echo "$SUBMISSION_DIR not exist, creating..."
    mkdir $SUBMISSION_DIR
fi

echo "Current directory: $(pwd)"

echo "Running prediction..."
python3 code/predict.py --song_dir $DATA_DIR/songs --lyric_dir $DATA_DIR/lyrics --output_dir $OUTPUT_DIR

