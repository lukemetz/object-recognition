#pragma once

#include <opencv2/opencv.hpp>
#include <memory>

struct Features
{
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


std::unique_ptr<Features> get_features(cv::Mat& src, int clip, int bins);
