#include <iostream>
#include <tuple>
#include <map>
#include <vector>
#include <string>
#include <opencv2/imgproc/imgproc.hpp>

struct Features;

typedef std::string Label;

class Classifier {
  public:
    Classifier();
    ~Classifier();

    void clear_probs();

    std::map<Label, std::map< std::tuple<int, int, int>, int> > feat_count;
    std::map<Label, int> labels_count;
    std::map<Label, std::map< std::tuple<int, int, int>, double> > feat_prob;
    void train_datum(const Features& datum, Label label);
    void calculate_probs(double smoothing);

    Label classify(const Features& datum);
    std::tuple<Label, double> classify_detailed(const Features& datum);
    
    static const std::vector<Label> labels;

    void locate_label(Label label, cv::Mat& src, cv::Mat& out, int clip);
};



