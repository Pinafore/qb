#include "clm.h"

void JelinekMercerFeature::set_interpolation(float lambda) {
  this->_bigram_contribution = 1.0 - lambda;
  this->_unigram_contribution = lambda;
}


float JelinekMercerFeature::interpolation() const {
    return this->_unigram_contribution;
}

void JelinekMercerFeature::set_slop(int slop) {
  assert(slop >= 0);
  _slop = slop;
}

void JelinekMercerFeature::set_censor_slop(bool censor_slop) {
  _censor_slop = censor_slop;
}

bool JelinekMercerFeature::censor_slop() const {
  return _censor_slop;
}

int JelinekMercerFeature::slop() const {
  return _slop;
}

void JelinekMercerFeature::set_cutoff(float cutoff) {
  _cutoff = cutoff;
}

float JelinekMercerFeature::cutoff() const {
  return _cutoff;
}

void JelinekMercerFeature::set_score(bool score) {
  _score = score;
}

bool JelinekMercerFeature::score() const {
  return _score;
}

void JelinekMercerFeature::set_log_length(bool log_length) {
  _log_length = log_length;
}

bool JelinekMercerFeature::log_length() const {
  return _log_length;
}

void JelinekMercerFeature::set_min_span(int span) {
  _min_span = span;
}

int JelinekMercerFeature::min_span() const {
  return _min_span;
}

void JelinekMercerFeature::set_min_start_rank(int rank) {
  _min_start_rank = rank;
}

int JelinekMercerFeature::min_start_rank() const {
  return _min_start_rank;
}

void JelinekMercerFeature::set_smooth(float smooth) {
  assert(smooth > 0.0);
  _smooth = smooth;
  _smooth_norm = smooth * (float)_vocab;
}

float JelinekMercerFeature::smooth() const {
  return _smooth;
}

void JelinekMercerFeature::add_stop(int word) {
  _stopwords.insert(word);
}

void JelinekMercerFeature::read_vocab(const std::string filename) {
  std::ifstream infile;
  infile.open(filename.c_str());

  int type;
  int count;
  int contexts;
  int next;
  std::string word;

  infile >> _corpora;

  infile >> _vocab;
  _types.resize(_vocab);
  _corpus_names.resize(_corpora);
  for (int ii=0; ii < _vocab; ++ii) {
    infile >> word;
    _types[ii] = word;
    if (ii % 25000 == 0)
      std::cout << "Read vocab " << word << " (" << ii << "/" << _vocab << ")" << std::endl;
  }

  this->_unigram.resize(_corpora);
  this->_bigram.resize(_corpora);
  this->_normalizer.resize(_corpora);
  this->_compare.resize(_corpora);

  /*
   * Size the counts appropriately
   */
  for (int cc=0; cc < _corpora; ++cc) {
    infile >> _corpus_names[cc];
    infile >> _compare[cc];
    _normalizer[cc] = 0;
  }
  std::cout << "Done reading " << _vocab << " vocab from " << _corpora << " corpora" << std::endl;
}

/*
 * Read in a protocol buffers contents and add to counts
 */
void JelinekMercerFeature::read_counts(const std::string filename) {
  std::cout << "reading corpus from " << filename << " (";
  std::ifstream infile;
  infile.open(filename.c_str());

  int corpus_id;
  std::string corpus_name;
  int num_contexts;

  infile >> corpus_name;
  infile >> corpus_id;
  infile >> num_contexts;
  std::cout << corpus_name << "," << corpus_id << ")" << std::endl;
  assert(_corpora > 0);

  assert(_normalizer[corpus_id] == 0.0);
  for (int vv=0; vv < num_contexts; ++vv) {
    // Set unigram counts
    int total;
    int first;
    int num_bigrams;
    std::string word;

    infile >> word;
    infile >> first;
    infile >> total;
    infile >> num_bigrams;
    _normalizer[corpus_id] += (float)total;
    _unigram[corpus_id][first] = total;
    assert(word == _types[first]);

    for (int bb=0; bb < num_bigrams; ++bb) {
      int second;
      int count;
      infile >> second;
      infile >> count;

      _bigram[corpus_id][bigram(first, second)] = count;
    }
  }
  infile.close();
}

int JelinekMercerFeature::bigram_count(int corpus, int first, int second) {
  bigram key = bigram(first, second);
  if (_bigram[corpus].find(key) == _bigram[corpus].end()) return 0;
  else return _bigram[corpus][key];
}

int JelinekMercerFeature::unigram_count(int corpus, int word) {
  if (_unigram[corpus].find(word) == _unigram[corpus].end()) return 0;
  else return _unigram[corpus][word];
}

int JelinekMercerFeature::sum(int *first, int nitems) {
    int i, sum = 0;
    for (i = 0; i < nitems; i++) {
        sum += first[i];
    }
    return sum;
}

const std::string JelinekMercerFeature::feature(const std::string name,
						int corpus, int *sent,
						int length) {
  assert(length > 0);
  assert(corpus < _corpora);
  int baseline = _compare[corpus];

  // Create vectors to hold probabilities and masks
  _slop_mask.resize(length);
  std::fill(_slop_mask.begin(), _slop_mask.end(), false);
  _span_mask.resize(length);
  _span_start.resize(length);
  _d_bigram.resize(length);
  _b_bigram.resize(length);
  _d_unigram.resize(length);
  _b_unigram.resize(length);

  // Step 1: find the spans
  // Consider the first word part of a span unless it's very frequent
  // This will also contribute to the overall likelihood computation later
  _span_mask[0] = true;

  // Extend from the first position.  Also compute the total likelihood while
  // we're at it.
  for (int ii=1; ii < length; ++ii) {
    if (this->bigram_count(corpus, sent[ii - 1], sent[ii]) > 0) {
      // The current word is only true if the previous word was or it isn't
      // too frequent.
      _span_mask[ii] = _span_mask[ii - 1] ||
          (sent[ii] >= _min_start_rank && _stopwords.find(sent[ii]) == _stopwords.end());
    } else {
      _span_mask[ii] = false;
    }
  }

  if (kDEBUG) {
    for (int ii=0; ii < length; ++ii) {
      if (_span_mask[ii]) std::cout << "\t" << "+" << _types[sent[ii]];
      else std::cout << "\t" << "-" << _types[sent[ii]];
    }
    std::cout << "\t<- tokens/mask" << std::endl;
  }

  // Filter spans that end with high-frequency words
  for (int ii=length; ii >= 0; --ii) {
    if (_span_mask[ii] && // It is in a span
	(ii==length || !_span_mask[ii+1]) && // at the end
	sent[ii] < _min_start_rank) {// and is high frequency
      _span_mask[ii] = false; // Then remove it from span
    }
  }

  // Add back in slop
  if (_slop > 0) {
    int position = 2;
    while (position < length - 1) {
      // Could this position expand a run if we had slop?
      if (_span_mask[position - 1] && !_span_mask[position]) {
        int slop_left = _slop;

        for (int ii=position; ii < std::min(position + _slop, length - 1); ++ii) {
          // If this is a LM hit, then
          // mark all positions before it as a slop position
          if (!_span_mask[ii]) {
            _slop_mask[ii] = true;
            --slop_left;
          }

          if (slop_left == 0) break;
          position = ii;
        }
        ++position;
      } else {
        // There's no run to expand, so just move onto the next position.
        ++position;
      }
    }
  }

  // For each element in the span, figure out where its span begins and compute
  // probabilities.
  _b_unigram[0] = this->score(baseline, -1, sent[0]);
  _d_unigram[0] = this->score(corpus, -1, sent[0]);
  float complete_prob = _d_unigram[0] - _b_unigram[0];

  int start = -1;
  for (int ii=0; ii < length; ++ii) {
    // Do we end a span?
    if (!(_span_mask[ii] || _slop_mask[ii])) {
      start = -1;
    } else {
      // Do we start a new one?
      if (start < 0) start = ii;
      // If we're in a span, we'll want unigram probabilities
      _d_unigram[ii] = this->score(corpus, -1, sent[ii]);
      _b_unigram[ii] = this->score(baseline, -1, sent[ii]);
    }
    _span_start[ii] = start;

    // save the probabilities for later span calculations
    _d_bigram[ii] = this->score(corpus, sent[ii - 1], sent[ii]);
    _b_bigram[ii] = this->score(baseline, sent[ii - 1], sent[ii]);
    complete_prob += _d_bigram[ii] - _b_bigram[ii];
  }

  if (kDEBUG) {
    std::cout << "\tX";
    for (int ii=1; ii < length; ++ii)
      std::cout << "\t" << this->bigram_count(corpus, sent[ii-1], sent[ii]);
    std::cout << std::endl;

    for (int ii=0; ii < length; ++ii) {
      if (_span_mask[ii]) std::cout << "\t" << "+";
      else std::cout << "\t" << "-";
    }
    std::cout << "\t<- revision" << std::endl;

    for (int ii=0; ii < length; ++ii) {
      if (_slop_mask[ii]) std::cout << "\t" << "+";
      else std::cout << "\t" << "-";
    }
    std::cout << "\t<- slop" << std::endl;

    for (int ii=0; ii < length; ++ii) std::cout << "\t" << _span_start[ii];
    std::cout << "\t<- start" << std::endl;

    for (int ii=0; ii < length; ++ii) std::cout << std::setprecision(3) << "\t" << _d_bigram[ii];
    std::cout << "\t<- domain bigram" << std::endl;

    for (int ii=0; ii < length; ++ii) std::cout << "\t" << std::setprecision(3) << _b_bigram[ii];
    std::cout << "\t<- base bigram" << std::endl;

    for (int ii=0; ii < length; ++ii) std::cout << "\t" << this->unigram_count(corpus, sent[ii]);
    std::cout << "\t<- domain unigram count" << std::endl;

    for (int ii=0; ii < length; ++ii) std::cout << "\t" << this->unigram_norm(corpus);
    std::cout << "\t<- domain unigram norm" << std::endl;

    for (int ii=0; ii < length; ++ii) std::cout << std::setprecision(3) << "\t" << _d_unigram[ii];
    std::cout << "\t<- domain unigram" << std::endl;

    for (int ii=0; ii < length; ++ii) std::cout << "\t" << std::setprecision(3) << _b_unigram[ii];
    std::cout << "\t<- base unigram" << std::endl;
  }

  // Now we can output the feature and compute the probability of spans
  std::ostringstream buffer;
  buffer << std::fixed;
  buffer << std::setprecision(2);
  int longest_span  = 0;
  float max_prob = 0;
  for (int ii=1; ii < length; ++ii) {
    if (_span_start[ii] >= 0) {
      if (ii - _span_start[ii] >= longest_span)
	longest_span = ii - _span_start[ii] + 1;

      // Skip if it's too short
      if (ii - _span_start[ii] < _min_span - 1) continue;

      for (int start = _span_start[ii]; start <= ii - _min_span; ++start) {
        float span_probability;
        if (_stopwords.find(sent[start]) != _stopwords.end()) continue;

        span_probability = _d_unigram[start] - _b_unigram[start];

        // First compute the span probabilities
        for (int jj = start + 1; jj <= ii; ++jj) {
          span_probability += _d_bigram[jj] - _b_bigram[jj];
        }

        if (span_probability > _cutoff) {
          if (span_probability > max_prob) max_prob = span_probability;
          buffer << corpus;
          for (int jj = start; jj <= ii; ++jj) {
            buffer << "_";
            if (_censor_slop && _slop_mask[jj]) buffer << "SLOP";
            else buffer << sent[jj];
          }
          if (_score) {
            buffer << ":";
            buffer << span_probability;
          }
          buffer << " ";
        }
      }
    }
  }

  // Output the probability
  if (_score) {
    buffer << name;
    buffer << "_PROB:";
    buffer << complete_prob / ((float)length);

    buffer << " ";
    buffer << name;
    buffer << "_MAX:";
    buffer << max_prob;
    buffer << " ";
  }

  buffer << name;
  buffer << "_LEN:";
  buffer << longest_span;

  if (_log_length) {
    buffer << " " << name;
    buffer << "_LGLEN:";
    buffer << log(1 + longest_span);
  }

  return buffer.str();
}

float JelinekMercerFeature::unigram_norm(int corpus) {
  return (float)this->_normalizer[corpus] + _smooth_norm;
}

float JelinekMercerFeature::score(int corpus, int first, int second) {
  // Store the unigram probability in the score
  float unigram_num = (float)this->unigram_count(corpus, second) + _smooth;
  float unigram_den = this->unigram_norm(corpus);
  float score = unigram_num / unigram_den;


  if (kDEBUG) {
    std::cout << _corpus_names[corpus] << " LL for " << first;
    //if (first >= 0) std::cout << "(" << _types[first] << ")";
    std::cout << " " << second << "(" << _types[second] << ")" << std::endl;
    std::cout << "UNI:" << unigram_num << "/" << unigram_den << "=" << score << std::endl;
  }

  if (first >= 0) { // first == -1 means we just want unigram probability
    int bigram_den = this->unigram_count(corpus, first);
    int bigram_num = this->bigram_count(corpus, first, second);
    if (bigram_den == 0 || bigram_num == 0) {
      score *= _unigram_contribution;
    } else {
      score = this->_bigram_contribution * (float)bigram_num / (float)bigram_den +
	_unigram_contribution * score;
    }
  }

  if (kDEBUG) {
    std::cout << "Final:" << log(score) << std::endl;
  }

  return log(score);
}
