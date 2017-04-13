source ~/qbenv
cd $QB_ROOT
nohup luigi --module qanta.pipeline.guesser AllGuessers &
sleep 5
