#!/bin/bash

# original script from: https://github.com/rsennrich/wmt16-scripts
# get En-Ro training data for WMT16

rm -rf data/wmt16_raw/train_corpus.en data/wmt16_raw/train_corpus.ro

if [ ! -f data/wmt16_raw/ro-en.tgz ];
then
  wget http://www.statmt.org/europarl/v7/ro-en.tgz -O data/wmt16_raw/ro-en.tgz
fi

if [ ! -f data/wmt16_raw/SETIMES2.ro-en.txt.zip ];
then
  wget http://opus.lingfil.uu.se/download.php?f=SETIMES2/en-ro.txt.zip -O data/wmt16_raw/SETIMES2.ro-en.txt.zip
fi

cd data/wmt16_raw
tar -xf ro-en.tgz
unzip SETIMES2.ro-en.txt.zip

cat europarl-v7.ro-en.en SETIMES.en-ro.en > train_corpus.en
cat europarl-v7.ro-en.ro SETIMES.en-ro.ro > train_corpus.ro

rm -rf europarl-v7.ro-en.en europarl-v7.ro-en.ro LICENSE README SETIMES.en-ro.en SETIMES.en-ro.ids SETIMES.en-ro.ro

cd ..
