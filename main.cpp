#include "feature.hpp"
#include "Classify.hpp"
#include <iostream>
#include <valarray>
#include <memory>
#include "glob.hpp"
#include <sstream>
#include <opencv2/imgproc/imgproc.hpp>
#include "TrainingHelper.hpp"

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
  
  vector<string> auto_bad = glob("training/auto/*.png");
  bad.insert(bad.end(), auto_bad.begin(), auto_bad.end());
  
  auto_bad = glob("training/auto2/*.png");
  bad.insert(bad.end(), auto_bad.begin(), auto_bad.end());
  
  auto_bad = glob("training/auto3/*.png");
  bad.insert(bad.end(), auto_bad.begin(), auto_bad.end());

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
  
  //TrainingHelper::train(argv[1], "dump", 60, true);
  //namedWindow("Classifier"); 
  std::vector<int> pixels = {40, 50, 60, 70, 80, 100};
  pixels = {50};
  Mat src = imread(argv[1]);
  Mat transcribe = imread(argv[1]);
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
          rectangle(transcribe, Point(x,y), Point(x+pixel, y+pixel), Scalar(128, 20, 30), 1);
        }
      }
    }
  }
  imwrite("out.png", transcribe);
  
}
