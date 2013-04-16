#include "feature.hpp"
#include <iostream>
#include <valarray>

#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
using namespace cv;


void gradiant(const Mat& src, Mat& mag, Mat& dir)
{
  for (int x=0; x < src.size().height; ++x) {
    for (int y=0; y < src.size().width; ++y) {
      Vec3b pix = src.at<Vec3b>(x,y);
      Vec3b pix_x0 = src.at<Vec3b>(x+1,y);
      Vec3b pix_x1 = src.at<Vec3b>(x-1,y);

      Vec3b pix_y0 = src.at<Vec3b>(x,y+1);
      Vec3b pix_y1 = src.at<Vec3b>(x,y-1);

      Vec3b gx = (pix_x0 - pix_x1)/2;
      char dx = (gx[0] + gx[1] + gx[2])/3; 
      
      Vec3b gy = (pix_y0 - pix_y1)/2;
      char dy = (gy[0] + gy[1] + gy[2])/3; 

      mag.at<unsigned char>(x,y) = dx + dy;
      dir.at<unsigned char>(x,y) = (3.1415 + atan2(dy, dx))/(2*3.1415)*255;
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
        dir.at<unsigned char>(x,y) = 1 + (dir.at<unsigned char>(x,y)/(256/bins));
      }
    }
  }
}

unique_ptr<Features> get_features(const Mat& src, int clip)
{
  Mat mag(src.size().height, src.size().width, CV_8UC1);
  Mat dir(src.size().height, src.size().width, CV_8UC1);
  gradiant(src, mag, dir);
  clip_mag_and_bin(clip, Features::bins, mag, dir);
  unique_ptr<Features> features (new Features(dir));
  return features; 
}

unique_ptr<Features> get_features(string filename, int clip)
{
  Mat src = imread(filename.c_str());
  Mat resized;
  resize(src, resized, Size(Features::width, Features::height));
  return get_features(resized, clip);
}
