#include "feature.hpp"
#include "Classify.hpp"
#include <iostream>
#include <valarray>
#include <memory>

#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
using namespace cv;

typedef std::map<Features, int> Type;
//label feature value probability
int main(int argc, char * argv[]) {
  Mat srcl = imread( argv[1] );
  Mat src;
  resize(srcl, src, Size(20, 20));

  auto features = get_features(src, 10, 8);
  Type t;
  t[*features] = 1;
  unique_ptr<Classifier> classifier(new Classifier());
  classifier->clear_probs();

  imwrite("out.jpg", features->pixels);
  cout << "Image found " << src.size() << "channels:" << src.channels() << endl;
}
