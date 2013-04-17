#include "TrainingHelper.hpp"
#include "feature.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <sstream>

using namespace std;
using namespace cv;

namespace TrainingHelper
{
  void train(std::string input, std::string dest, int size, bool all_false)
  {
    Mat src = imread(input);
    int width = src.size().width;
    int height = src.size().height;
    int i = 0;
    for (int x = 0; x < width - size; x+=size/4) {
      for (int y = 0; y  < height - size; y+=size/4) {
        Mat snip = src(Rect(x, y, size, size));
        char key = 'a';
        if (all_false == false) {
          imshow("Classify", snip);
          key = waitKey(0);
          cout << key << endl;
        }
        stringstream s;
        s << dest;
        string ds(1, key);
        s << "/" << ds << "/";
        s << i++ << ".png";
        cout << s.str() << endl;
        imwrite(s.str(), snip);
        }
    }
  }
}
