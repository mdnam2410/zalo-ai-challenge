import argparse
import json
import os

from tqdm.auto import tqdm


def labels_to_lyrics(label_dir, lyric_dir):
    """Extract lyrics from label files and save them to corresponding text file."""
    if not os.path.exists(lyric_dir):
        os.makedirs(lyric_dir)

    label_files = [file for file in os.listdir(label_dir) if file.endswith(".json")]
    for label_file in tqdm(label_files):
        with open(os.path.join(label_dir, label_file), "r") as f:
            label = json.load(f)

        lyric = []
        for sentence in label:
            for word in sentence["l"]:
                lyric.append(word["d"])
        with open(
            os.path.join(lyric_dir, label_file.replace(".json", ".txt")), "w"
        ) as f:
            f.write(" ".join(lyric))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label_dir",
        type=str,
        required=True,
        help="Path to label directory containing json files.",
    )
    parser.add_argument(
        "--lyric_dir",
        type=str,
        required=True,
        help="Path to lyric directory to save text files.",
    )
    args = parser.parse_args()

    labels_to_lyrics(args.label_dir, args.lyric_dir)
