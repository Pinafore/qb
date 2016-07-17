clm_wrap.cxx: clm.swig
	swig -c++ -python $<

clm_wrap.o: clm_wrap.cxx
	gcc -O3 `python3-config --includes` -std=c++11 -fPIC -c $< -o $@

clm.o: clm.cpp clm.h
	gcc -O3 `python3-config --includes` -std=c++11 -fPIC -c $< -o $@

_clm.so: clm.o clm_wrap.o
	g++ -shared `python3-config --ldflags` $^ -o $@

clm: _clm.so