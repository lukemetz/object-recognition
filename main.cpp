#include "feature.hpp"
#include "Classify.hpp"
#include <iostream>
#include <valarray>
#include <memory>
#include "glob.hpp"

#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
using namespace cv;

int main(int argc, char * argv[]) {
  unique_ptr<Classifier> classifier(new Classifier());
  classifier->clear_probs();
  
  vector<string> good = glob("training/good/*.png");
  for (auto file : good) {
    auto features = get_features(file, 10);
    classifier->train_datum(*features, "Ball");
  }
  
  vector<string> bad = glob("training/bad/*.png");
  for (auto file : bad) {
    auto features = get_features(file, 10);
    classifier->train_datum(*features, "Nothing");
  }

  classifier->calculate_probs(1); 

  int correct =  0;
  int wrong = 0;
  for (auto file : good) {
    auto features = get_features(file, 10);
    Label l = classifier->classify(*features);
    if (l == "Ball") {
      correct += 1;
    } else {
      wrong += 1;
    }
  }
  
  for (auto file : bad) {
    auto features = get_features(file, 10);
    Label l = classifier->classify(*features);
    std::cout << l <<endl;
    if (l == "Nothing") {
      correct += 1;
    } else {
      wrong += 1;
    }
  }
  
  cout << "Correct:" << correct << "Wrong:" << wrong << endl;
}
