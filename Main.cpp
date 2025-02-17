#include "Network.hpp"
#include <vector>
#include <cmath>

static float ActivationSigmoid(float x)
{
    return 1.f / (1.f + std::exp(-x));
}

static float DerivativeActivationSigmoid(float output)
{
    return output * (1.f - output);
}

static float LossMSE(int n, float pred, float expected)
{
    return (2.f / static_cast<float>(n)) * (pred - expected);
}


int main() {
    std::vector<std::vector<std::vector<float>>> weights {
        {{0.8f, 0.3f}, {0.4f, 0.2f}}, 
        {{0.6f, 0.8f}, {0.9f, 0.1f}}
    };

    Network nn(2, ActivationSigmoid, DerivativeActivationSigmoid, LossMSE, weights);

    std::vector<float> inputs   = {0.1f, 0.5f};
    std::vector<float> expected = {0.05f, 0.95f};
    nn.Fit(inputs, expected);

    auto preds = nn.Predict(inputs);
    for (size_t i = 0; i < preds.size(); i++) {
        std::printf("Prediction output %d before fitting %f\n", i, preds[i]);
    }

    for (size_t i = 0; i < 1000; i++) {
        nn.Fit(inputs, expected);
    }

    preds = nn.Predict(inputs);
    for (size_t i = 0; i < preds.size(); i++) {
        std::printf("Prediction output %d after fitting %f\n", i, preds[i]);
    }

    return 0;
}