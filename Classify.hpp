#include <iostream>
struct Features;

enum class Label {
  Nothing, Ball
};

class Classifier {
  public:
    Classifier();
    ~Classifier();
    int smoothing = 1;

    void train_datum(Features * datum, Label label);
    Label classify(Features * datum);
};
