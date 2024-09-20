# Running the buzzer expo interface

## Setup
Data files go in the folder `qanta/expo/questions_do_not_upload`.
- Download and unzip the folder `questions_do_not_upload` from this Drive folder: https://drive.google.com/drive/u/0/folders/1SZWePX1PNjOZ8A_bbwwLZrp_ZVsykl61 (request access if needed).
- Place it in `qanta/expo`. **Do not upload questions to GitHub. Questions should stay in questions_do_not_upload or the Drive.**
- Download Tom in VoiceOver Utility: https://discussions.apple.com/thread/254892791. If you can't (e.g. don't have a Mac), comment out these lines in answer():
`os.system("afplay /System/Library/Sounds/Glass.aiff")
os.system("say -v Tom %s" % ans.replace("'", "").split("(")[0])``

## Commands
- From inside `qanta/expo`, run the buzzer system:
`python3 buzzer.py --questions [QUESTION FILE] --model_directory [MODEL FOLDER] --model [MODEL] --answer_equivalents [EQUIVALENTS FILE]`

Example:
`python3 buzzer.py --questions questions_do_not_upload/packet_1.csv --model_directory questions_do_not_upload/2024_gpr_co --model packet_1 --answer_equivalents questions_do_not_upload/2024_gpr_co/equivalents_new.json --skip 1 --human 10 --computer 20`

- Buzzer check:
When "Player 1, please buzz in" appears, press `1` on the keyboard.
When "QANTA says" appears, press `Ctrl + X` on the keyboard.

- Press any key to print the next word of the question and to reveal the answer once the model has buzzed.

- Skipping ahead or correcting the score: use `skip [NUMBER OF QUESTIONS] --human [HUMAN SCORE] --computer [COMPUTER SCORE]`, e.g.:
`python3 buzzer.py --questions questions_do_not_upload/2024_co/packet_1.csv --model_directory questions_do_not_upload/2024_gpr_co --model packet_1 --answer_equivalents questions_do_not_upload/2024_co/2024_gpr_co/equivalents_new.json --skip 1 --human 10 --computer 20`
