# rm data/expo_guess.db
# python extract_expo_features.py
# python util/reweight_labels.py features/expo/word.label.feat
# python util/reweight_labels.py features/dev/sentence.label.feat
for WEIGHT in 8 16 32 64
do
    echo $WEIGHT
    # paste -d' ' features/dev/sentence.label.$WEIGHT `python feature_config.py` | gzip > features/dev/sentence.$WEIGHT.vw_input
    vw --compressed -d features/dev/sentence.$WEIGHT.vw_input --early_terminate 100 -k -b 28 --loss_function logistic -f models/sentence.full.$WEIGHT.vw -q gt -q gd --l1 0.000001
    # paste -d' ' features/expo/word.label.$WEIGHT `python feature_config.py --fold=expo --granularity=word` | gzip > features/expo/expo.$WEIGHT.vw_input
    vw --compressed -t -d features/expo/expo.$WEIGHT.vw_input -i models/sentence.full.$WEIGHT.vw  -p results/expo/expo.$WEIGHT.pred --audit > results/expo/expo.$WEIGHT.audit
    python reporting/evaluate_predictions.py --buzzes=results/expo/expo.$WEIGHT.buzz --qbdb=data/questions.db --question_out='' --meta=features/expo/word.meta --perf=results/expo/word.$WEIGHT.full.perf --neg_weight=$WEIGHT.000000 --vw_config=full --expo=data/expo.csv --question_out=results/expo/readable.txt --finals=results/expo/expo.$WEIGHT.final --pred=results/expo/expo.$WEIGHT.pred
done


for WEIGHT in 8 16 32 64
do
    echo $WEIGHT
    grep ",True" results/expo/word.$WEIGHT.full.perf | wc
done
