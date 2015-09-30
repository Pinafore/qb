#ifndef EXAMPLE_H
#define EXAMPLE_H

#include <string>
#include <vector>
#include <map>
#include <math.h>
#include <cassert>
#include <utility>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <sstream>

typedef std::pair <int, int> bigram;
#define kDEBUG false
#define kMIN_SPAN 1
#define kSMOOTH 1e-3
#define kMIN_RATIO 2.0
#define kSTART_RANK_MIN 100 // Don't start spans with too high a probability

class JelinekMercerFeature {
 public:
    float interpolation() const;
    void set_interpolation(float lambda);
    void read_counts(const std::string filename);
    int sum(int *first, int nitems);
    int bigram_count(int corpus, int first, int second);
    int unigram_count(int corpus, int word);
    float score(int corpus, int first, int second);
    const std::string feature(const std::string name, int corpus, 
			      int *sent, int length);
 private:
    float _bigram_contribution;
    float _unigram_contribution;
    int _corpora;
    int _vocab;
    float _mean;
    float _std;
    std::vector<float>                   _normalizer;
    std::vector< std::map<bigram, int> > _bigram;
    std::vector< std::vector<int> >      _unigram;
    std::vector<std::string>             _types;
    std::vector<bool>                    _span_mask;
    std::vector<int>                     _span_start;
    std::vector<std::string>             _corpus_names;
    std::vector<int>                     _compare;
    std::vector<float> _d_bigram;
    std::vector<float> _b_bigram;
    std::vector<float> _d_unigram;
    std::vector<float> _b_unigram;
};

#endif
