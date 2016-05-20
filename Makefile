data/deep/glove.840B.300d.txt.gz:
	mkdir -p data/deep
	curl http://www-nlp.stanford.edu/data/glove.840B.300d.txt.gz > $@

data/deep/params: data/deep/glove.840B.300d.txt.gz
	python3 guesser/util/format_dan.py --database=data/questions.db --threshold=5
	python3 guesser/util/load_embeddings.py
	python3 guesser/dan.py

data/kenlm.binary:
	mkdir -p temp
	python3 cli.py build_mentions_lm_data data/wikipedia temp/wiki_sent
	lmplz -o 5 < temp/wiki_sent > data/kenlm.arpa
	build_binary data/kenlm.arpa $@
	rm temp/wiki_sent

data/classifier/category.pkl:
	mkdir -p data/classifier
	python3 util/classifier.py --attribute=category

data/classifier/ans_type.pkl:
	mkdir -p data/classifier
	python3 util/classifier.py --attribute=ans_type

data/classifier/gender.pkl:
	mkdir -p data/classifier
	python3 util/classifier.py --attribute=gender


classifier: data/classifier/category.pkl data/classifier/ans_type.pkl data/classifier/gender.pkl 

data/wikifier/data/input/: util/wikification.py
	rm -rf $@
	mkdir -p $@
	python3 util/wikification.py

data/wikifier/data/output/: data/wikifier/data/input
	rm -rf $@
	mkdir -p $@
	cp lib/wikifier-3.0-jar-with-dependencies.jar data/wikifier/wikifier-3.0-jar-with-dependencies.jar
	cp lib/STAND_ALONE_GUROBI.xml data/wikifier/STAND_ALONE_GUROBI.xml
	(cd data/wikifier && java -Xmx10G -jar wikifier-3.0-jar-with-dependencies.jar -annotateData data/input data/output false STAND_ALONE_GUROBI.xml)

clm/clm_wrap.cxx: clm/clm.swig
	swig -c++ -python $<

clm/clm_wrap.o: clm/clm_wrap.cxx
	gcc -O3 `python3-config --includes` -fPIC -c $< -o $@

clm/clm.o: clm/clm.cpp clm/clm.h
	gcc -O3 `python3-config --includes` -fPIC -c $< -o $@

clm/_clm.so: clm/clm.o clm/clm_wrap.o
	g++ -shared `python3-config --ldflags` $^ -o $@

prereqs: data/wikifier/data/output data/kenlm.binary data/deep/params classifier
