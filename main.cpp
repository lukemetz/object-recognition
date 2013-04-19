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
  
  //TrainingHelper::train(argv[1], "dump", 40, true);
  //namedWindow("Classifier"); 
  std::vector<int> pixels = {40, 50, 60, 70, 80, 100};
  Mat src = imread(argv[1]);
  Mat transcribe = imread(argv[1]);
  int width = src.size().width;
  int height = src.size().height;
  for (int pixel : pixels) {
    for (int x = 0; x < width - pixel; x+=10) {
      for (int y = 0; y  < height - pixel; y+=10) {
        Mat nsrc = src(Rect(x, y, pixel, pixel));
        auto features = get_features(nsrc, clip);
        auto feat = classifier->classify_detailed(*features);
        Label l = get<0>(feat);
        double prob = get<1>(feat);
        
        if (l == "Ball" && prob > -600) {
          circle(transcribe, Point(x+pixel/2, y+pixel/2), pixel/2, Scalar(0, 0, 0));
       
        }

        if (l == "Ball" && prob > - 520) {
          cout << x << ", " << y << "Size:" <<  pixel << "prob" << prob << endl;
          //rectangle(transcribe, Point(x,y), Point(x+pixel, y+pixel), Scalar(128, 20, 30), 1);
          int color = prob + 550; 
          color = color - 255;
          circle(transcribe, Point(x+pixel/2, y+pixel/2), pixel/2, Scalar(2*color, 255, 0), 2);
        }
      }
    }
  }
  imshow("result2", src);
  imshow("result", transcribe);

  waitKey(0);
  imwrite("out.png", transcribe);
  
}
