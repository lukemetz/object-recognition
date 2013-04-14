#include <iostream>
#include <tuple>
#include <map>
#include <vector>

struct Features;

enum class Label {
  Nothing, Ball
};


class Classifier {
  public:
    Classifier();
    ~Classifier();

    void clear_probs();

    std::map<Label, std::map< std::tuple<int, int, int>, double> > feat_count;
    std::map<Label, int> labels_count;

    std::map<Label, std::map< std::tuple<int, int, int>, double> > feat_prob;

    void train_datum(const Features& datum, Label label);
    void calculate_probs(int smoothing);

    Label classify(const Features& datum);
    
    static const std::vector<Label> labels;
};

