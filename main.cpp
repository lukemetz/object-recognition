#include "feature.hpp"
#include "Classify.hpp"
#include <iostream>
#include <valarray>
#include <memory>
#include "glob.hpp"
#include <sstream>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

int main(int argc, char * argv[]) {
  int clip = 5;

  unique_ptr<Classifier> classifier(new Classifier());
  classifier->clear_probs();
  
  vector<string> good = glob("training/good/*.png");
  for (auto file : good) {
    auto features = get_features(file, clip);
    classifier->train_datum(*features, "Ball");
  }
  
  vector<string> bad = glob("training/bad/*.png");
  for (auto file : bad) {
    auto features = get_features(file, clip);
    classifier->train_datum(*features, "Nothing");
  }

  classifier->calculate_probs(1); 

  int correct =  0;
  int wrong = 0;
  for (auto file : good) {
    auto features = get_features(file, clip);
    Label l = classifier->classify(*features);
    if (l == "Ball") {
      correct += 1;
    } else {
      cout << file << "Classified as: " << l << "Now in" << wrong << endl;
      wrong += 1;
    }
  }
  
  for (auto file : bad) {
    auto features = get_features(file, clip);
    Label l = classifier->classify(*features);
    if (l == "Nothing") {
      correct += 1;
    } else {
      cout << file << "Classified as: " << l << "Now in" << wrong << endl;
      wrong += 1;
    }
  }
  cout << "Correct:" << correct << "Wrong:" << wrong << endl;

  std::vector<int> pixels = {40, 50, 60, 70, 80, 100};
  Mat src = imread(argv[1]);
  int width = src.size().width;
  int height = src.size().height;
  for (int pixel : pixels) {
    for (int x = 0; x < width - pixel; x+=10) {
      for (int y = 0; y  < height - pixel; y+=10) {
        Mat nsrc = src(Rect(x, y, pixel, pixel));
        auto features = get_features(nsrc, clip);
        Label l = classifier->classify(*features);
        if (l == "Ball") {
          cout << x << ", " << y << "Size:" << pixel << endl;
        }
      }
    }
  }
  
}
