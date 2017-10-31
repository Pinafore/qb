#!/usr/bin/env bash


cd $1
java -Xmx10g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLP -threads 1 -annotators tokenize,ssplit,pos,parse -ssplit,pos,parse -ssplit.eolonly -file $2 -outputFormat text -parse.model edu/stanford/nlp/models/srparser/englishSR.ser.gz -outputDirectory $3
