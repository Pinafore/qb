clean:
	rm -f clm/clm_wrap.cxx
	rm -f clm/_ctrie.so

clm/ctrie_wrap.cxx: clm/ctrie.swig
	swig -c++ -python $<

clm/ctrie_wrap.o: clm/ctrie_wrap.cxx
	gcc -O3 `python3-config --includes` -std=c++11 -fPIC -c $< -o $@

clm/clm.o: clm/clm.cpp clm/clm.h
	gcc -O3 `python3-config --includes` -std=c++11 -fPIC -c $< -o $@

clm/_ctrie.so: clm/clm.o clm/ctrie_wrap.o
	g++ -shared `python3-config --ldflags` $^ -o $@
	touch clm/_SUCCESS

clm: clm/_ctrie.so
