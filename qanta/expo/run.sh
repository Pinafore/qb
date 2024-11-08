model=gpt4o
packet=3
room=242
round=1
python buzzer.py --question  ./../../data/questions_do_not_upload/oct_data/packet_output_csvs/packet_"$packet"_output.csv \
--model_directory ./../../data/questions_do_not_upload/oct_packets/oct_packets_buzzpts_"$model" \
--model packet_"$packet"_output \
--answer_equivalents ./../../data/questions_do_not_upload/oct_data/oct_equivalents.json \
--output room"$room"_round"$round".csv \
--skip 1 \
--players 4

