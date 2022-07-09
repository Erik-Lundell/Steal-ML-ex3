#!/bin/sh

[[ -f train ]] && [[ -f test ]] || {
    echo "train and test don't exist";
    exit;
}

svm-scale -l -1 -u 1 -s range -y -1 1 train > train.scale
svm-scale -r range test > test.scale

# get eval
svm-train train.scale
svm-predict test.scale train.scale.model test.pred > 'eval'

# clean up
rm -f test.pred
rm -f range
