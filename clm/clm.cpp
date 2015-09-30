#include "clm.h"

float JelinekMercerFeature::interpolation() const {
    return this->_unigram_contribution;
}
void JelinekMercerFeature::set_interpolation(float lambda) {
  this->_bigram_contribution = 1.0 - lambda;
  this->_unigram_contribution = lambda;
}
void JelinekMercerFeature::read_counts(const std::string filename) {
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

  /*
   * Size the counts appropriately
   */
  assert(_corpora > 0);
  this->_unigram.resize(_corpora);
  this->_bigram.resize(_corpora);
  this->_normalizer.resize(_corpora);
  this->_compare.resize(_corpora);

  for (int cc=0; cc < _corpora; ++cc) {
    infile >> _corpus_names[cc];
    infile >> _compare[cc];
    if (cc % 1000 == 0)
      std::cout << "Read corpus for " << _corpus_names[cc] << " (" << cc << "/" << _corpora << ")" << std::endl;
    _unigram[cc].resize(_vocab);
    _normalizer[cc] = 0;

    for (int vv=0; vv < _vocab; ++vv) {
      infile >> type;
      infile >> count;
      infile >> contexts;
      assert(type == vv);

      // Set unigram counts
      _normalizer[cc] += (float)count + kSMOOTH;
      _unigram[cc][type] = count;

      for (int bb=0; bb<contexts; ++bb) {
        infile >> next;
        infile >> count;
        _bigram[cc][bigram(type, next)] = count;
      }
    }
  }
  std::cout << "Done reading corpus" << std::endl;
}

int JelinekMercerFeature::bigram_count(int corpus, int first, int second) {
  bigram key = bigram(first, second);
  if (_bigram[corpus].find(key) == _bigram[corpus].end()) return 0;
  else return _bigram[corpus][key];
}

int JelinekMercerFeature::unigram_count(int corpus, int word) {
  return _unigram[corpus][word];
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

  // Create vectors to hold probabilities
  _span_mask.resize(length);
  _span_start.resize(length);
  _d_bigram.resize(length);
  _b_bigram.resize(length);
  _d_unigram.resize(length);
  _b_unigram.resize(length);

  // Step 1: find the spans
  // Consider the first word part of a span unless it's very frequent
  // This will also contribute to the overall likelihood computation later
  _b_unigram[0] = this->score(baseline, -1, sent[0]);
  _d_unigram[0] = this->score(corpus, -1, sent[0]);
  _span_mask[0] = sent[0] >= kSTART_RANK_MIN;

  // Extend from the first position.  Also compute the total likelihood while
  // we're at it.
  float complete_prob = _d_unigram[0] - _b_unigram[0];
  for (int ii=1; ii < length; ++ii) {
    if (this->bigram_count(corpus, sent[ii - 1], sent[ii]) > 0) {
      // The current word is only true if the previous word was or it isn't
      // too frequent.
      _span_mask[ii] = _span_mask[ii - 1] || (sent[ii] >= kSTART_RANK_MIN);
      
      // The previous mask is only true if it was already or the following
      // word isn't too frequent.
      _span_mask[ii - 1] = _span_mask[ii - 1] || (_span_mask[ii] && (sent[ii-1] >= kSTART_RANK_MIN));
    } else {
      _span_mask[ii] = false;
    }

    // save the probabilities for later span calculations
    _d_bigram[ii] = this->score(corpus, sent[ii - 1], sent[ii]);
    _b_bigram[ii] = this->score(baseline, sent[ii - 1], sent[ii]);
    complete_prob += _d_bigram[ii] - _b_bigram[ii];
  }

  if (kDEBUG) {
    for (int ii=0; ii < length; ++ii) {
      if (_span_mask[ii]) std::cout << "\t" << "+" << _types[sent[ii]];
      else std::cout << "\t" << "-" << _types[sent[ii]];
    }
    std::cout << "\t<- mask" << std::endl;
  }

  // Filter spans that end with high-frequency words
  for (int ii=length; ii >= 0; --ii) {
    if (_span_mask[ii] && // It is in a span
	(ii==length || !_span_mask[ii+1]) && // at the end
	sent[ii] < kSTART_RANK_MIN) {// and is high frequency
      _span_mask[ii] = false; // Then remove it from span
    }
  }

  // For each element in the span, figure out where its span begins and filter
  // out low probability starts.
  int start = -1;
  for (int ii=0; ii < length; ++ii) {
    // See if we start a new span
    if (_span_mask[ii] && start < 0) {
      // We'll need the probability later, so go ahead and comput it now.
      _b_unigram[ii] = this->score(baseline, -1, sent[ii]);
      start = ii;
    } else if (!_span_mask[ii]) {
      start = -1;
    }
    _span_start[ii] = start;
  }

  if (kDEBUG) {
    std::cout << "\tX";
    for (int ii=1; ii < length; ++ii)
      std::cout << "\t" << this->bigram_count(corpus, sent[ii-1], sent[ii]);
    std::cout << std::endl;


    for (int ii=0; ii < length; ++ii) {
      if (_span_mask[ii]) std::cout << "\t" << "+" << sent[ii];
      else std::cout << "\t" << "-" << sent[ii];
    }
    std::cout << "\t<- revision" << std::endl;

    for (int ii=0; ii < length; ++ii) std::cout << "\t" << _span_start[ii];
    std::cout << "\t<- start" << std::endl;

    for (int ii=0; ii < length; ++ii) std::cout << std::setprecision(3) << "\t" << _d_bigram[ii];
    std::cout << "\t<- domain bigram" << std::endl;

    for (int ii=0; ii < length; ++ii) std::cout << "\t" << std::setprecision(3) << _b_bigram[ii];
    std::cout << "\t<- base bigram" << std::endl;

    for (int ii=0; ii < length; ++ii) std::cout << std::setprecision(3) << "\t" << _d_unigram[ii];
    std::cout << "\t<- domain unigram" << std::endl;

    for (int ii=0; ii < length; ++ii) std::cout << "\t" << std::setprecision(3) << _b_unigram[ii];
    std::cout << "\t<- base unigram" << std::endl;
  }

  // Now we can output the feature and compute the probability of spans
  std::ostringstream buffer;
  int longest_span  = 0;
  float max_prob = 0;
  for (int ii=kMIN_SPAN; ii < length; ++ii) {
    if (_span_start[ii] >= 0) {
      if (ii - _span_start[ii] > longest_span) 
	longest_span = ii - _span_start[ii] + 1;

      // Skip if it's too short
      if (ii - _span_start[ii] < kMIN_SPAN) continue;

      float span_probability;
      int start;
      if (_span_start[ii] == ii - kMIN_SPAN) {
        start = _span_start[ii];
        span_probability = _d_unigram[start] - _b_unigram[start];
      } else {
        start = ii - kMIN_SPAN;
        span_probability = _d_bigram[start] - _b_bigram[start];
      }

      // First compute the span probabilities
      for (int jj = start + 1; jj <= ii; ++jj) {
        span_probability += _d_bigram[ii] - _b_bigram[ii];
      }

      if (span_probability > kMIN_RATIO) {
        if (span_probability > max_prob) max_prob = span_probability;
        buffer << _corpus_names[corpus];
        for (int jj = start; jj <= ii; ++jj) {
          buffer << "_";
          buffer << _types[sent[jj]];
        }
        buffer << ":";
        buffer << span_probability;
        buffer << " ";
      }
    }
  }

  // Output the probability
  buffer << name;
  buffer << "_PROB:";
  buffer << complete_prob / ((float)length);

  buffer << " ";
  buffer << name;
  buffer << "_MAX:";
  buffer << max_prob;

  buffer << " ";
  buffer << name;
  buffer << "_LEN:";
  buffer << log(1 + longest_span);

  return buffer.str();
}

float JelinekMercerFeature::score(int corpus, int first, int second) {
  // Store the unigram probability in the score
  float unigram_num = (float)this->unigram_count(corpus, second) + kSMOOTH;
  float unigram_den = (float)this->_normalizer[corpus];
  float score = unigram_num / unigram_den;

  if (kDEBUG) {
    std::cout << "LL for " << first << " " << second << std::endl;
    std::cout << "UNI:" << unigram_num << "/" << unigram_den << "=" << score << std::endl;
  }

  if (first >= 0) { // first == -1 means we just want unigram probability
    float bigram_den = (float)this->unigram_count(corpus, first);
    float bigram_num = (float)this->bigram_count(corpus, first, second);
    if (bigram_den == 0 || bigram_num == 0) {
      score *= _unigram_contribution;
    } else {
      score = this->_bigram_contribution * bigram_num / bigram_den +
	_unigram_contribution * score;
    }
  }

  if (kDEBUG) {
    std::cout << "Final:" << log(score) << std::endl;
  }

  return log(score);
}
