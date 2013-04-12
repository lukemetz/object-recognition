#include <iostream>
#include <opencv2/opencv.hpp>
#include <valarray>

#include <opencv2/imgproc/imgproc.hpp>
using namespace std;



int main(int argc, char * argv[]) {
  cv::Mat src;
  src = cv::imread( argv[1] );
  cv::Mat mag(src.size().height, src.size().width, CV_8UC1);
  cv::Mat dir(src.size().height, src.size().width, CV_8UC1);

  cout << "Image found " << src.size() << "channels:" << src.channels() << endl;
  for (int x=0; x < src.size().width; ++x) {
    for (int y=0; y < src.size().height; ++y) {
      cv::Vec3b pix = src.at<cv::Vec3b>(x,y);
      //Vec3b pix = src.at<cv::Vec3b>(x,y);
      cv::Vec3b pix_x0 = src.at<cv::Vec3b>(x+1,y);
      cv::Vec3b pix_x1 = src.at<cv::Vec3b>(x-1,y);

      cv::Vec3b pix_y0 = src.at<cv::Vec3b>(x,y+1);
      cv::Vec3b pix_y1 = src.at<cv::Vec3b>(x,y-1);

      cv::Vec3b gx = (pix_x0 - pix_x1)/2;
      char dx = (gx[0] + gx[1] + gx[2])/3; 
      
      cv::Vec3b gy = (pix_y0 - pix_y1)/2;
      char dy = (gy[0] + gy[1] + gy[2])/3; 

      mag.at<unsigned char>(x,y) = dx + dy;
      dir.at<unsigned char>(x,y) = (3.1415 + atan2(dy, dx))/(2*3.1415)*255;

    }
  }
  cv::imwrite("out.jpg", dir);
}
