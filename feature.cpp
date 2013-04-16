#include "feature.hpp"
#include <iostream>
#include <valarray>

#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
using namespace cv;


void gradiant(const Mat& src, Mat& mag, Mat& dir)
{
  for (int x=1; x < src.size().height-1; ++x) {
    for (int y=1; y < src.size().width-1; ++y) {
      Vec3b pix = src.at<Vec3b>(x,y);
      Vec3b pix_x0 = src.at<Vec3b>(x+1,y);
      Vec3b pix_x1 = src.at<Vec3b>(x-1,y);

      Vec3b pix_y0 = src.at<Vec3b>(x,y+1);
      Vec3b pix_y1 = src.at<Vec3b>(x,y-1);

      Vec3i gx = (Vec3i(pix_x0) - Vec3i(pix_x1))/2;
      int dx = (gx[0] + gx[1] + gx[2])/3; 
      
      Vec3i gy = (Vec3i(pix_y0) - Vec3i(pix_y1))/2;
      int dy = (gy[0] + gy[1] + gy[2])/3; //Sum abs values?? take max?

      mag.at<unsigned char>(x-1,y-1) = (unsigned char) (dx + dy)/2+ 128;
      if (dx == 0 && dy == 0) {
        dx = 12;
        dy = 123;
      }
      dir.at<unsigned char>(x-1,y-1) =(unsigned char) (3.1415 + atan2(dy, dx))/(2*3.1415)*256;
    }
  }
}

void clip_mag_and_bin(int amount, int bins, Mat& mag, Mat& dir) {
  for (int x=0; x < mag.size().height; ++x) {
    for (int y=0; y < mag.size().width; ++y) {
      unsigned char current_mag = mag.at<unsigned char>(x,y);
      if (current_mag < amount) {
        mag.at<unsigned char>(x,y) = 0;
        dir.at<unsigned char>(x,y) = 0;
      } else {
        dir.at<unsigned char>(x,y) = 1 + (dir.at<unsigned char>(x,y)/(256.0/bins));
      }
    }
  }
}

unique_ptr<Features> get_features(const Mat& src, int clip)
{
  Mat resized;
  resize(src, resized, Size(Features::width+2, Features::height+2));

  Mat mag(resized.size().height, resized.size().width, CV_8UC1);
  Mat dir(resized.size().height, resized.size().width, CV_8UC1);
  gradiant(resized, mag, dir);
  clip_mag_and_bin(clip, Features::bins, mag, dir);
  unique_ptr<Features> features (new Features(dir));
  //unique_ptr<Features> features (new Features(dir));
  features->b4_filter = resized;
  return features; 
}

unique_ptr<Features> get_features(string filename, int clip)
{
  Mat src = imread(filename.c_str());
  return get_features(src, clip);
}
