import argparse
import json
import os
from dataclasses import dataclass
from typing import List

import torch
import torchaudio
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCTC, AutoProcessor


class Wav2Vec2Aligner:
    def __init__(self, model_path, cuda):
        self.cuda = cuda
        self.config = AutoConfig.from_pretrained(model_path)
        self.model = AutoModelForCTC.from_pretrained(model_path)
        self.model.eval()
        if self.cuda:
            self.model.to(device="cuda")
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.resampler = torchaudio.transforms.Resample(44000, 16000)
        blank_id = 0
        vocab = list(self.processor.tokenizer.get_vocab().keys())
        for i in range(len(vocab)):
            if vocab[i] == "[PAD]" or vocab[i] == "<pad>":
                blank_id = i
        print("Blank Token id [PAD]/<pad>", blank_id)
        self.blank_id = blank_id

    def speech_file_to_array_fn(self, wav_path):
        # TODO: resample from diff freq
        speech_array, sampling_rate = torchaudio.load(wav_path)
        self.resampler.orig_freq = sampling_rate
        speech = self.resampler(speech_array).squeeze().numpy()
        return speech

    def align_single_sample(self, item):
        blank_id = self.blank_id
        transcript = "|".join(item["sent"])
        if not os.path.isfile(item["wav_path"]):
            print(item["wav_path"], "not found in wavs directory")

        speech_array = self.speech_file_to_array_fn(item["wav_path"])
        speech_array = torch.from_numpy(speech_array).float()
        # convert an audio signal to mono by averaging samples across channels.
        if speech_array.shape[0] > 1:
            speech_array = torch.mean(speech_array, dim=0)

        inputs = self.processor(
            speech_array, sampling_rate=16_000, return_tensors="pt", padding=True
        )

        if self.cuda:
            inputs = inputs.to(device="cuda")

        with torch.no_grad():
            logits = self.model(inputs.input_values).logits

        # get the emission probability at frame level
        emissions = torch.log_softmax(logits, dim=-1)
        emission = emissions[0].cpu().detach()

        # get labels from vocab
        # labels = ([""] + list(self.processor.tokenizer.get_vocab().keys()))[
        #     :-1
        # ]  # logits don't align with the tokenizer's vocab
        # labels = list(self.processor.tokenizer.get_vocab().keys())

        # dictionary = {c: i for i, c in enumerate(labels)}
        # # TODO: len tokens != len transcript
        # tokens = []
        # for c in transcript:
        #     c = c.lower()
        #     if c in dictionary:
        #         tokens.append(dictionary[c])

        tokens = self.processor(text=transcript, return_tensors="pt").input_ids[0]

        def get_trellis(emission, tokens, blank_id=0):
            """
            Build a trellis matrix of shape (num_frames + 1, num_tokens + 1)
            that represents the probabilities of each source token being at a certain time step
            """
            num_frames = emission.size(0)
            num_tokens = len(tokens)

            # Trellis has extra dimensions for both time axis and tokens.
            # The extra dim for tokens represents <SoS> (start-of-sentence)
            # The extra dim for time axis is for simplification of the code.
            trellis = torch.full((num_frames + 1, num_tokens + 1), -float("inf"))
            trellis[:, 0] = 0
            for t in range(num_frames):
                trellis[t + 1, 1:] = torch.maximum(
                    # Score for staying at the same token
                    trellis[t, 1:] + emission[t, blank_id],
                    # Score for changing to the next token
                    trellis[t, :-1] + emission[t, tokens],
                )
            return trellis

        trellis = get_trellis(emission, tokens, blank_id)

        @dataclass
        class Point:
            token_index: int
            time_index: int
            score: float

        def backtrack(trellis, emission, tokens, blank_id=0):
            """
            Walk backwards from the last (sentence_token, time_step) pair to build the optimal sequence alignment path
            """
            # Note:
            # j and t are indices for trellis, which has extra dimensions
            # for time and tokens at the beginning.
            # When referring to time frame index `T` in trellis,
            # the corresponding index in emission is `T-1`.
            # Similarly, when referring to token index `J` in trellis,
            # the corresponding index in transcript is `J-1`.
            j = trellis.size(1) - 1
            t_start = torch.argmax(trellis[:, j]).item()

            path = []
            for t in range(t_start, 0, -1):
                # 1. Figure out if the current position was stay or change
                # Note (again):
                # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
                # Score for token staying the same from time frame J-1 to T.
                stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
                # Score for token changing from C-1 at T-1 to J at T.
                changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

                # 2. Store the path with frame-wise probability.
                prob = (
                    emission[t - 1, tokens[j - 1] if changed > stayed else 0]
                    .exp()
                    .item()
                )
                # Return token index and time index in non-trellis coordinate.
                path.append(Point(j - 1, t - 1, prob))

                # 3. Update the token
                if changed > stayed:
                    j -= 1
                    if j == 0:
                        break
            else:
                raise ValueError("Failed to align")
            return path[::-1]

        path = backtrack(trellis, emission, tokens, blank_id)

        @dataclass
        class Segment:
            label: str
            start: int
            end: int
            score: float

            def __repr__(self):
                return (
                    f"{self.label}\t{self.score:4.2f}\t{self.start:5d}\t{self.end:5d}"
                )

            def __dict__(self):
                return {
                    "s": self.start,
                    "e": self.end,
                    "d": self.label,
                }

            @property
            def length(self):
                return self.end - self.start

        def merge_repeats(path):
            """
            Merge repeated tokens into a single segment. Note: this shouldn't affect repeated characters from the
            original sentences (e.g. `ll` in `hello`)
            """
            i1, i2 = 0, 0
            segments = []
            while i1 < len(path):
                while i2 < len(path) and path[i1].token_index == path[i2].token_index:
                    i2 += 1
                score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
                segments.append(
                    Segment(
                        transcript[path[i1].token_index],
                        path[i1].time_index,
                        path[i2 - 1].time_index + 1,
                        score,
                    )
                )
                i1 = i2
            return segments

        segments = merge_repeats(path)
        # Merge words
        def merge_words(segments, separator="|"):
            words = []
            i1, i2 = 0, 0
            while i1 < len(segments):
                if i2 >= len(segments) or segments[i2].label == separator:
                    if i1 != i2:
                        segs = segments[i1:i2]
                        word = "".join([seg.label for seg in segs])
                        score = sum(seg.score * seg.length for seg in segs) / sum(
                            seg.length for seg in segs
                        )
                        # convert to ms (1 frame = 20ms)
                        words.append(
                            Segment(
                                word,
                                20 * segments[i1].start,
                                20 * segments[i2 - 1].end,
                                score,
                            )
                        )
                    i1 = i2 + 1
                    i2 = i1
                else:
                    i2 += 1
            return words

        word_segments = merge_words(segments)

        @dataclass
        class Sentence:
            start: int
            end: int
            segments: List[Segment]

            # def __repr__(self):
            #     return f"{self.label}\t{self.score:4.2f}\t{self.start*20:5d}\t{self.end*20:5d}"

            def __dict__(self):
                # TODO: *320/16 -> ms
                return {
                    "s": self.start,
                    "e": self.end,
                    "l": [seg.__dict__() for seg in self.segments],
                }

            @property
            def length(self):
                return self.end - self.start

        def merge_sentence(segments, num_words_per_sentence):
            # merge words into sentences
            sentences = []
            start = 0
            end = 0
            for num_words in num_words_per_sentence:
                end += num_words
                start_time = segments[start].start
                end_time = segments[end - 1].end
                sentences.append(Sentence(start_time, end_time, segments[start:end]))
                start = end
            return sentences

        num_words = item["num_words"]
        sentences = merge_sentence(word_segments, num_words)

        with open(item["out_path"], "w", encoding="utf8") as file:
            json.dump([s.__dict__() for s in sentences], file, ensure_ascii=False)

    def align_data(self, song_dir, lyric_dir, output_dir):

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        items = []
        song_files = os.listdir(song_dir)

        for song_file in song_files:
            # if no lyrics, skip
            filename = song_file.split(".")[0]
            lyric_file = os.path.join(lyric_dir, filename + ".json")
            if not os.path.exists(lyric_file):
                continue

            with open(lyric_file, "r") as f:
                label = json.load(f)

            lyric = []
            num_words_per_sentence = []
            for sentence in label:
                num_words = 0
                for word in sentence["l"]:
                    lyric.append(word["d"])
                    num_words += 1
                num_words_per_sentence.append(num_words)
            items.append(
                {
                    "sent": lyric,
                    "num_words": num_words_per_sentence,
                    "wav_path": os.path.join(song_dir, song_file),
                    "out_path": os.path.join(output_dir, filename + ".json"),
                }
            )

        for item in tqdm(items):
            self.align_single_sample(item)

        # # load text file
        # lines = open(text_file, encoding="utf8").readlines()

        # items = []
        # for line in lines:
        #     if len(line.strip().split("\t")) != 2:
        #         print("Script must be in format: 00001  this is my sentence")
        #         exit()

        #     wav_name, sentence = line.strip().split("\t")
        #     wav_path = os.path.join(wav_dir, wav_name + ".wav")
        #     out_path = os.path.join(output_dir, wav_name + ".json")

        #     items.append({"sent": sentence, "wav_path": wav_path, "out_path": out_path})
        # print("Number of samples found in script file", len(items))

        # for item in tqdm(items):
        #     self.align_single_sample(item)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        default="not-tanh/wav2vec2-large-xlsr-53-vietnamese",
        help="pretrained model path",
    )
    parser.add_argument(
        "--song_dir",
        type=str,
        help="directory containing wavs",
        default="/Users/leminhhin/Downloads/public_test/vocals",
    )
    parser.add_argument(
        "--lyric_dir",
        type=str,
        help="directory containing text",
        default="data/public_test/lyrics",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="output directory containing the alignment files",
        default="submission/run_5",
    )
    parser.add_argument("--cuda", action="store_true")

    args = parser.parse_args()

    aligner = Wav2Vec2Aligner(args.model_path, args.cuda)
    aligner.align_data(args.song_dir, args.lyric_dir, args.output_dir)


if __name__ == "__main__":
    main()
