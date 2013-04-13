#include "Classify.hpp"
#include "feature.hpp"

Classifier::Classifier()
{
}

Classifier::~Classifier()
{
}

void Classifier::train_datum(Features * datum, Label label)
{
  
}

Label Classifier::classify(Features * datum)
{
  return Label::Nothing;
}
