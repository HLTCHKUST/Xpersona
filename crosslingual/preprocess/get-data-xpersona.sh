set -e

# data path
MAIN_PATH=$PWD
DATA_PATH=$PWD/data
PERSONA_PATH=$DATA_PATH/xpersona
PROCESSED_PATH=$DATA_PATH/processed/XNLG/eval/xpersona
PREPROCESS=$MAIN_PATH/get_binary_data.py
# CODES_PATH=$DATA_PATH/codes_xnli_15
# VOCAB_PATH=$DATA_PATH/vocab_xnli_15
CODES_PATH=$DATA_PATH/codes_xnli_100   # for languages based on XLM-R
VOCAB_PATH=$DATA_PATH/vocab_xnli_100   # for languages based on XLM-R

# tools paths
TOOLS_PATH=$PWD/tools
TOKENIZE=$TOOLS_PATH/tokenize.sh
LOWER_REMOVE_ACCENT=$TOOLS_PATH/lowercase_and_remove_accent.py
FASTBPE=$TOOLS_PATH/fastBPE/fast

mkdir -p $PERSONA_PATH
mkdir -p $PROCESSED_PATH

for split in train valid test; do
  # for lang in en zh fr; do
  # for lang in zh fr; do
  for lang in en id it jp ko; do
    for seg in x y; do
      $FASTBPE applybpe $PROCESSED_PATH/$split.$seg.$lang $PERSONA_PATH/$split.$seg.$lang $CODES_PATH
      python $PREPROCESS $VOCAB_PATH $PROCESSED_PATH/$split.$seg.$lang
    done
  done
done
