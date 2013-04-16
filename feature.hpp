#pragma once

#include <opencv2/opencv.hpp>
#include <memory>
#include <string>

struct Features
{
  static const int width = 20;
  static const int height = 20;
  static const int bins = 8;

  cv::Mat pixels;
  Features(cv::Mat mat) {
    pixels = mat;
  };
};

inline bool operator <(const Features a, const Features b) {
  auto iter2 = a.pixels.begin<unsigned char>();
  for (auto iter = b.pixels.begin<unsigned char>(); iter != b.pixels.end<unsigned char>(); ++iter, ++iter2) {
    if (*iter != *iter2) {
      return *iter < *iter2;
    }
  }
  return false;
}
//takes resized image in src, and returns features with clip
std::unique_ptr<Features> get_features(const cv::Mat& src, int clip);
//Gets the image at filename, resizes and calculates features
std::unique_ptr<Features> get_features(std::string filename, int clip);
