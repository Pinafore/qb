This file documents how to run the expo in an older version of qanta

## Expo Instructions

The expo files can be generated from a completed qanta run by calling

```bash
luigi --module qanta.expo.pipeline --workers 2 AllExpo
```

If that has already been done you can restore the expo files from a backup instead of running the
pipeline

```bash
./checkpoint restore expo
```

Then to finally run the expo

```bash
python3 qanta/expo/buzzer.py --questions=output/expo/test.questions.csv --buzzes=output/expo/test.16.buzz --output=output/expo/competition.csv --finals=output/expo/test.16.final
```
