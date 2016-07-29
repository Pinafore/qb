data/external/deep/glove.840B.300d.txt:
	mkdir -p data/external/deep
	curl http://nlp.stanford.edu/data/glove.840B.300d.zip > /tmp/glove.840B.300d.zip
	unzip /tmp/glove.840B.300d.zip -d data/external/deep
	rm /tmp/glove.840B.300d.zip

data/external/wikipedia/:
	mkdir -p $@
	python3 cli.py init_wiki_cache $@

output/kenlm.binary: data/external/wikipedia clm
	mkdir -p temp
	mkdir -p output
	python3 setup.py download
	python3 cli.py build_mentions_lm_data data/external/wikipedia /tmp/wiki_sent
	lmplz -o 5 < /tmp/wiki_sent > temp/kenlm.arpa
	build_binary temp/kenlm.arpa $@
	rm /tmp/wiki_sent temp/kenlm.arpa

data/external/wikifier/input/: clm
	rm -rf $@
	mkdir -p $@
	python3 cli.py wikify data/external/wikifier/input/

data/external/wikifier/output/: data/external/wikifier/input
	rm -rf $@
	mkdir -p $@
	(cd data/external/Wikifier2013 && java -Xmx10G -jar dist/wikifier-3.0-jar-with-dependencies.jar -annotateData ../wikifier/input ../wikifier/output false configs/STAND_ALONE_NO_INFERENCE.xml)

clm/clm_wrap.cxx: clm/clm.swig
	swig -c++ -python $<

clm/clm_wrap.o: clm/clm_wrap.cxx
	gcc -O3 `python3-config --includes` -std=c++11 -fPIC -c $< -o $@

clm/clm.o: clm/clm.cpp clm/clm.h
	gcc -O3 `python3-config --includes` -std=c++11 -fPIC -c $< -o $@

clm/_clm.so: clm/clm.o clm/clm_wrap.o
	g++ -shared `python3-config --ldflags` $^ -o $@

clm: clm/_clm.so

prereqs: data/external/wikifier/output output/kenlm.binary
