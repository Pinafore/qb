#ifndef CLM_H
#define CLM_H

#include <string>
#include <vector>
#include <unordered_set>
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

class JelinekMercerFeature {
 public:
    float interpolation() const;
    void set_jm_interpolation(float lambda);
    void set_slop(int slop);
    void set_censor_slop(bool censor_slop);
    bool censor_slop() const;
    int slop() const;
    void set_log_length(bool log_length);
    bool log_length() const;
    void set_score(bool score);
    bool score() const;
    void add_stop(int word);
    void set_unigram_smooth(float smooth);
    float smooth() const;
    void set_min_start_rank(int rank);
    int min_start_rank() const;
    void set_cutoff(float cutoff);
    float cutoff() const;
    void set_min_span(int span);
    int min_span() const;
    void set_max_span(int span);
    int max_span() const;
    void read_vocab(const std::string filename);
    void read_counts(const std::string filename);
    int sum(int *first, int nitems);
    int bigram_count(int corpus, int first, int second);
    int unigram_count(int corpus, int word);
    float unigram_norm(int corpus);
    float score(int corpus, int first, int second);
    const std::string feature(const std::string name, int corpus,
			      int *sent, int length);
 private:
    float _bigram_contribution;
    float _unigram_contribution;
    int _corpora;
    int _vocab;
    float _cutoff;
    int _slop;
    int _min_span;
    int _max_span;
    bool _score;
    bool _censor_slop;
    bool _log_length;
    float _smooth;
    float _smooth_norm;
    float _mean;
    int _min_start_rank;
    std::vector<float>                   _normalizer;
    std::vector< std::map<bigram, int> > _bigram;
    std::vector< std::map<int, int> >    _unigram;
    std::vector<std::string>             _types;
    std::vector<bool>                    _slop_mask;
    std::vector<bool>                    _span_mask;
    std::vector<int>                     _span_start;
    std::vector<std::string>             _corpus_names;
    std::vector<int>                     _compare;
    std::vector<float>                   _d_bigram;
    std::vector<float>                   _b_bigram;
    std::vector<float>                   _d_unigram;
    std::vector<float>                   _b_unigram;
    std::unordered_set<int>              _stopwords;
};

#endif
