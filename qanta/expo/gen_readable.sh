model=mistral
players=0
skip_id=21  # so it will skip all questions

# python3 buzzer.py --answer_equivalents ./../../data/questions_do_not_upload/nov_equivalents.json --question ./../../data/questions_do_not_upload/nov_packets/packet_output_csvs/packet_11_output.csv --model_directory ./../../data/questions_do_not_upload/nov_packets/nov_packets_buzzpts_mistral --model packet_11_output  --output eve_debug.csv

for packet in {3..12};
do
    # Notes: path might be different for people :)
    python buzzer.py \
        --questions questions_do_not_upload/2024_online/nov_data/packet_"$packet"_output.csv \
        --model_directory questions_do_not_upload/2024_online/nov_packets/nov_packets_buzzpts_"$model" \
        --model packet_"$packet"_output \
        --answer_equivalents questions_do_not_upload/2024_online/nov_equivalents.json \
        --readable questions_do_not_upload/2024_online/readable/readable_packet_"$packet"_"$model".txt \
        --players "$players" \
        --skip $skip_id \
        --output ""
done