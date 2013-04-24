#include "Classify.hpp"
#include "feature.hpp"
#include <cmath>

const std::vector<Label> Classifier::labels = {"Nothing", "Ball"};

using namespace std;
using namespace cv;

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
  return get<0>(classify_detailed(datum));
}

tuple<Label, double> Classifier::classify_detailed(const Features& datum)
{
  double prior = 1;
  map<Label, double> probs;

  for (Label label : Classifier::labels) {
    probs[label] = prior;
    for (int x = 0; x < Features::width; ++x) {
      for (int y = 0; y < Features::height; ++y) {
        int bin = datum.pixels.at<unsigned char>(y, x);
        probs[label] += log( feat_prob[label][tuple<int, int, int>(x, y, bin)] );
      }
    }
  }
  
  auto max_iter = max_element(probs.begin(), probs.end(),
    [](const pair<Label, double>& p1, const pair<Label, double>& p2) {
      return p1.second < p2.second;
    });

  return tuple<Label, double> (max_iter->first, max_iter->second);

}

void Classifier::locate_label(Label label, cv::Mat& src, cv::Mat& out, int clip)
{
  std::vector<int> pixels = {40, 50, 60, 70, 80, 100};
  out = src.clone(); 
  int width = src.size().width;
  int height = src.size().height;
  for (int pixel : pixels) {
    for (int x = 0; x < width - pixel; x+=10) {
      for (int y = 0; y  < height - pixel; y+=10) {
        Mat nsrc = src(Rect(x, y, pixel, pixel));
        auto features = get_features(nsrc, clip);
        auto feat = classify_detailed(*features);
        Label l = get<0>(feat);
        double prob = get<1>(feat);

        if (l == label && prob > -600) {
          circle(out, Point(x+pixel/2, y+pixel/2), pixel/2, Scalar(100, 100, 100));
        }
        
        if (l == label && prob > - 520) {
          cout << x << ", " << y << "Size:" <<  pixel << "prob" << prob << endl;
          //rectangle(transcribe, Point(x,y), Point(x+pixel, y+pixel), Scalar(128, 20, 30), 1);
          int color = prob + 550; 
          color = color - 255;
          circle(out, Point(x+pixel/2, y+pixel/2), pixel/2, Scalar(0, 0, 0), 2);
        } 
      }
    }
  }
}
