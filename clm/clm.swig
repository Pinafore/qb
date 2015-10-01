%module clm
%include "std_string.i"
%include "carrays.i"
%array_class(int, intArray);
%{
    #define SWIG_FILE_WITH_INIT
    #include "clm.h"
%}
%include "clm.h"