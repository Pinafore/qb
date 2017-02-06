source ~/qbenv
cd $QB_ROOT
nohup luigi --module qanta.pipeline.dan RunTFDanExperiment &
sleep 5
