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
  
  //Assemble the training data 
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
  for (int i =0; i < 2; i ++) { 
    auto_bad = glob("training/auto/*.png");
    bad.insert(bad.end(), auto_bad.begin(), auto_bad.end());
    auto_bad = glob("training/auto4/*.png");
    bad.insert(bad.end(), auto_bad.begin(), auto_bad.end());
    auto_bad = glob("training/auto5/*.png");
    bad.insert(bad.end(), auto_bad.begin(), auto_bad.end());
    auto_bad = glob("training/auto6/*.png");
    bad.insert(bad.end(), auto_bad.begin(), auto_bad.end());
    auto_bad = glob("training/auto7/*.png");
    bad.insert(bad.end(), auto_bad.begin(), auto_bad.end());
  }
  for (auto file : bad) {
    auto features = get_features(file, clip);
    classifier->train_datum(*features, "Nothing");
  }

  classifier->calculate_probs(1); 

  //Go through already classified data to check classifier.
  //TODO Add an evaluation training set
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
  cout << "Percent: " << (float)correct/(correct + wrong) << endl;
  
  //TrainingHelper::train(argv[1], "dump", 40, true);
  
  Mat src = imread(argv[1]);
  Mat out;
  classifier->locate_label("Ball", src, out, clip);

  imshow("result2", src);
  imshow("result", out);

  waitKey(0);
  imwrite("out.png", out);
  
}
