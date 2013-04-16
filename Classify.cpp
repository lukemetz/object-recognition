#include "Classify.hpp"
#include "feature.hpp"
#include <cmath>

const std::vector<Label> Classifier::labels = {"Nothing", "Ball"};

using namespace std;

Classifier::Classifier()
{

}

Classifier::~Classifier()
{
}

void Classifier::clear_probs()
{
  for (int x = 0; x < Features::width; ++x) {
    for (int y = 0; y < Features::height; ++y) {
      for (int bin = 0; bin < Features::bins+1; ++bin) {
        for (auto label : Classifier::labels) {
          feat_count[label][tuple<int, int, int>(x, y, bin)] = 0;
          feat_prob[label][tuple<int, int, int>(x, y, bin)] = 0;
          labels_count[label] = 0;
        }
      }
    }
  }
}

void Classifier::train_datum(const Features& datum, Label label)
{
  labels_count[label] += 1;
  for (int x = 0; x < Features::width; ++x) {
    for (int y = 0; y < Features::height; ++y) {
      int bin = datum.pixels.at<unsigned char> (y, x);
      feat_count[label][tuple<int, int, int>(x, y, bin)] += 1;
    }
  }
}

void Classifier::calculate_probs(double smoothing)
{
  for (int x = 0; x < Features::width; ++x) {
    for (int y = 0; y < Features::height; ++y) {
      for (int bin = 0; bin < Features::bins+1; ++bin) {
        for (Label label : Classifier::labels) {
          tuple<int, int, int> feat(x, y, bin);
          double prob_there = (feat_count[label][feat] + smoothing)/
            (labels_count[label] + smoothing);
          feat_prob[label][feat] = prob_there;
        }
      }
    }
  }
}

Label Classifier::classify(const Features& datum)
{
  double prior = 1;
  map<Label, double> probs;
  for (Label label : Classifier::labels) {
    probs[label] = prior;
    for (int x = 0; x < Features::width; ++x) {
      for (int y = 0; y < Features::height; ++y) {
        int bin = datum.pixels.at<unsigned char>(y, x);
        probs[label] += log( feat_prob[label][tuple<int, int, int>(x, y, bin)] );
        cout << feat_prob[label][tuple<int, int, int>(x, y, bin)] << endl;
        if (feat_prob[label][tuple<int, int, int>(x, y, bin)] == 0) {
          std::cout << x << "," << y << "," << bin << label << endl;
          //std::cout << x << "," << y << "," << bin << (label == Label::Nothing ? "Nothing" : "Something") << endl;
        }
      }
    }
  }
  
  auto max_iter = max_element(probs.begin(), probs.end(),
    [](const pair<Label, double>& p1, const pair<Label, double>& p2) {
      return p1.second < p2.second;
    });
  //cout << probs[Label::Nothing] << "," << probs[Label::Ball] << endl;
  return max_iter->first;
}
