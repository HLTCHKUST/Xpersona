# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Tokenize text data in various languages
# Usage: e.g.   cat wiki.ar | tokenize.sh ar

set -e

# N_THREADS=8

# lg=$1
# TOOLS_PATH=$PWD/tools

# # moses
# MOSES=$TOOLS_PATH/mosesdecoder
# REPLACE_UNICODE_PUNCT=$MOSES/scripts/tokenizer/replace-unicode-punctuation.perl
# NORM_PUNC=$MOSES/scripts/tokenizer/normalize-punctuation.perl
# REM_NON_PRINT_CHAR=$MOSES/scripts/tokenizer/remove-non-printing-char.perl
# TOKENIZER=$MOSES/scripts/tokenizer/tokenizer.perl

# # Chinese
# if [ "$lg" = "zh" ]; then
#   $TOOLS_PATH/stanford-segmenter-*/segment.sh pku /dev/stdin UTF-8 0 | $REPLACE_UNICODE_PUNCT | $NORM_PUNC -l $lg | $REM_NON_PRINT_CHAR
# # Thai
# elif [ "$lg" = "th" ]; then
#   cat - | $REPLACE_UNICODE_PUNCT | $NORM_PUNC -l $lg | $REM_NON_PRINT_CHAR | python $TOOLS_PATH/segment_th.py
# # Japanese
# elif [ "$lg" = "ja" ]; then
#   cat - | $REPLACE_UNICODE_PUNCT | $NORM_PUNC -l $lg | $REM_NON_PRINT_CHAR | kytea -notags
# # other languages
# else
#   cat - | $REPLACE_UNICODE_PUNCT | $NORM_PUNC -l $lg | $REM_NON_PRINT_CHAR | $TOKENIZER -no-escape -threads $N_THREADS -l $lg
# fi

echo "train.x.zh"
bash /home/zihan/stanford-segmenter-2018-10-16/segment.sh ctb train.x.Zh UTF-8 0 > train.x.zh

echo "train.y.zh"
bash /home/zihan/stanford-segmenter-2018-10-16/segment.sh ctb train.y.Zh UTF-8 0 > train.y.zh

echo "valid.x.zh"
bash /home/zihan/stanford-segmenter-2018-10-16/segment.sh ctb valid.x.Zh UTF-8 0 > valid.x.zh

echo "valid.y.zh"
bash /home/zihan/stanford-segmenter-2018-10-16/segment.sh ctb valid.y.Zh UTF-8 0 > valid.y.zh

echo "test.x.zh"
bash /home/zihan/stanford-segmenter-2018-10-16/segment.sh ctb test.x.Zh UTF-8 0 > test.x.zh

echo "test.y.zh"
bash /home/zihan/stanford-segmenter-2018-10-16/segment.sh ctb test.y.Zh UTF-8 0 > test.y.zh
