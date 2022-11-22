SUBMISSION_DIR="/result"
DATA_DIR="/data"

echo "Loading data from $DATA_DIR"

# Create submission directory if not exist
if [ ! -d $SUBMISSION_DIR ]; then
    echo "$SUBMISSION_DIR not exist, creating..."
    mkdir $SUBMISSION_DIR
fi

echo "Saving results to $SUBMISSION_DIR/submission.zip"
touch "$SUBMISSION_DIR/submission.zip"

