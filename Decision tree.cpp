#include <mlpack/core.hpp>
#include <mlpack/core/data/load.hpp>
#include <mlpack/methods/decision_tree/decision_tree.hpp>

using namespace mlpack;

int main()
{
    // Load the Bank Marketing dataset (replace 'your_dataset.csv' with the actual file)
    arma::mat data;
    data::Load("your_dataset.csv", data, true);

    // Extract features and labels
    arma::rowvec labels = data.row(data.n_rows - 1);
    arma::mat predictors = data.rows(0, data.n_rows - 2);

    // Train a decision tree classifier
    mlpack::tree::DecisionTree<> decisionTree(predictors, labels, 2);

    // Predictions for a new data point (replace 'new_data_point' with actual data)
    arma::rowvec new_data_point = ...;
    double prediction = decisionTree.Classify(new_data_point);

    // Output the prediction
    std::cout << "Predicted class: " << prediction << std::endl;

    return 0;
}
