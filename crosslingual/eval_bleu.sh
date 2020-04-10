
set -e

MOSES_PATH=$PWD/tools/mosesdecoder
MULTIBLEU=$MOSES_PATH/scripts/generic/multi-bleu.perl

# French
PATH_MODEL=$PWD/dump/xpersona/ftOnZh        # path of the saved model folder
PATH_OUTPUT=$PATH_MODEL/output_0_zh.txt     # path of the model output
PATH_REF=$PATH_MODEL/ref_0_zh.txt           # path of the reference

echo "Evaluating translations..."
$MULTIBLEU $PATH_REF < $PATH_OUTPUT > $PATH_MODEL/eval.bleu

cat $PATH_MODEL/eval.bleu
