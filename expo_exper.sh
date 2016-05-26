rm data/expo_guess.db
python extract_expo_features.py
python util/reweight_labels.py features/expo/word.label.feat
for WEIGHT in 8 16 32 64
do
    echo $WEIGHT
    paste -d' ' features/expo/word.label.$WEIGHT features/expo/word.lm.feat features/expo/word.deep.feat features/expo/word.answer_present.feat features/expo/word.wikilinks.feat| gzip > features/expo/expo.$WEIGHT.vw_input
    paste -d' ' features/dev/sentence.label.$WEIGHT features/dev/sentence.lm.feat features/dev/sentence.deep.feat features/dev/sentence.answer_present.feat features/dev/sentence.wikilinks.feat | gzip > features/dev/sentence.$WEIGHT.vw_input
    vw --compressed -d features/dev/sentence.$WEIGHT.vw_input --early_terminate 100 -k -q gt -q gd -b 28 --loss_function logistic -f models/sentence.full.$WEIGHT.vw 
    vw --compressed -t -d features/expo/expo.$WEIGHT.vw_input -i models/sentence.full.$WEIGHT.vw  -p results/expo/expo.$WEIGHT.pred --audit > results/expo/expo.$WEIGHT.audit
    python reporting/evaluate_predictions.py --buzzes=results/expo/expo.$WEIGHT.buzz --qbdb=data/questions.db --question_out='' --meta=features/expo/word.meta --perf=results/expo/word.$WEIGHT.full.perf --neg_weight=$WEIGHT.000000 --vw_config=full --expo=data/expo.csv --question_out=results/expo/readable.txt --finals=results/expo/expo.$WEIGHT.final --pred=results/expo/expo.$WEIGHT.pred
done


for WEIGHT in 8 16 32 64
do
    echo $WEIGHT
    grep ",True" results/expo/word.$WEIGHT.full.perf | wc
done
