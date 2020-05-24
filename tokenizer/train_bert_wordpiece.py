import argparse
import glob

from tokenizers import BertWordPieceTokenizer

parser = argparse.ArgumentParser()
parser.add_argument(
    "--files",
    default='./train_en.txt',
    metavar="path",
    type=str,
    required=False,
    help="The files to use as training; accept '**/*.txt' type of patterns \
                          if enclosed in quotes",
)
parser.add_argument(
    "--out",
    default="./vocab_files/",
    type=str,
    help="Path to the output directory, where the files will be saved",
)
parser.add_argument(
    "--name", default="bert-wordpiece-en", type=str, help="The name of the output vocab files"
)
args = parser.parse_args()

files = glob.glob(args.files)
if not files:
    print(f"File does not exist: {args.files}")
    exit(1)


# Initialize an empty tokenizer
tokenizer = BertWordPieceTokenizer(
    clean_text=True, strip_accents=True, lowercase=True,
)

# And then train
tokenizer.train(
    files,
    vocab_size=20000,
    min_frequency=2,
    show_progress=True,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    limit_alphabet=1000,
    wordpieces_prefix="##",
)

# Save the files
tokenizer.save(args.out, args.name)
