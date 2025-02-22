#include "pch.h"

#include <cmath>
#include <vector>
#include <gtest/gtest.h>
#include "../Network.hpp"
#include "../nnMath.hpp"

// Helper functions for tests
float TestActivationSigmoid(float x)
{
    return 1.f / (1.f + std::exp(-x));
}

float TestDerivativeActivationSigmoid(float output)
{
    return output * (1.f - output);
}

float TestLossMSE(int n, float pred, float expected)
{
    return (2.f / static_cast<float>(n)) * (pred - expected);
}

// Test fixture for Network tests
class NetworkTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create a simple network with known weights for testing
        std::vector<std::vector<std::vector<float>>> weights{
            {{0.8f, 0.3f}, {0.4f, 0.2f}},
            {{0.6f, 0.8f}, {0.9f, 0.1f}}
        };
        network = new Network(2, TestActivationSigmoid, TestDerivativeActivationSigmoid, TestLossMSE, weights);
    }

    void TearDown() override
    {
        delete network;
    }

    Network* network;
};


namespace NetworkTests
{

// Test activation function
TEST(ActivationTest, SigmoidFunction)
{
    float input = 0.0f;
    float expected = 0.5f;
    EXPECT_NEAR(TestActivationSigmoid(input), expected, 1e-6);

    input = 1.0f;
    expected = 0.7310586f;
    EXPECT_NEAR(TestActivationSigmoid(input), expected, 1e-6);
}

// Test derivative of activation function
TEST(ActivationTest, DerivativeSigmoidFunction)
{
    float output = 0.5f;
    float expected = 0.25f;  // 0.5 * (1 - 0.5)
    EXPECT_NEAR(TestDerivativeActivationSigmoid(output), expected, 1e-6);
}

// Test loss function
TEST(LossTest, MeanSquaredError)
{
    int n = 2;
    float pred = 0.7f;
    float expected = 0.5f;
    float expectedLoss = (2.0f / 2.0f) * (0.7f - 0.5f);
    EXPECT_NEAR(TestLossMSE(n, pred, expected), expectedLoss, 1e-6);
}

// Test forward pass
TEST_F(NetworkTest, ForwardPass)
{
    std::vector<float> inputs = { 0.1f, 0.5f };
    auto predictions = network->Predict(inputs);

    // With the given weights and inputs, we can calculate expected outputs
    // First layer:
    // Neuron 1: 0.1 * 0.8 + 0.5 * 0.3 = 0.23
    // Neuron 2: 0.1 * 0.4 + 0.5 * 0.2 = 0.14
    // After sigmoid activation
    float expected_layer1_1 = TestActivationSigmoid(0.23f);
    float expected_layer1_2 = TestActivationSigmoid(0.14f);

    // Second layer (using outputs from first layer)
    // Neuron 1: expected_layer1_1 * 0.6 + expected_layer1_2 * 0.8
    // Neuron 2: expected_layer1_1 * 0.9 + expected_layer1_2 * 0.1

    EXPECT_EQ(predictions.size(), 2);
    EXPECT_NEAR(predictions[0], 0.681854f, 1e-5);
    EXPECT_NEAR(predictions[1], 0.635299f, 1e-5);
}

// Test backpropagation single step
TEST_F(NetworkTest, BackpropagationChangesWeights)
{
    std::vector<float> inputs = { 0.1f, 0.5f };
    std::vector<float> expected = { 0.05f, 0.95f };

    // Store initial weights
    auto initial_weights = network->CopyWeights();

    // Perform single training step
    network->Fit(inputs, expected);

    // Get updated weights
    auto updated_weights = network->CopyWeights();

    // Verify weights have changed
    EXPECT_NE(initial_weights, updated_weights);
}

// Test convergence
TEST_F(NetworkTest, Convergence)
{
    std::vector<float> inputs = { 0.1f, 0.5f };
    std::vector<float> expected = { 0.05f, 0.95f };

    // Train for multiple iterations
    for (int i = 0; i < 1000; i++) {
        network->Fit(inputs, expected);
    }

    auto final_predictions = network->Predict(inputs);

    // Check if predictions are closer to expected values
    EXPECT_NEAR(final_predictions[0], expected[0], 0.1);
    EXPECT_NEAR(final_predictions[1], expected[1], 0.1);
}

// Test nnMath functions
TEST(MathTest, DotProduct)
{
    std::vector<float> v1 = { 1.0f, 2.0f, 3.0f };
    std::vector<float> v2 = { 4.0f, 5.0f, 6.0f };
    float expected = 32.0f;  // 1*4 + 2*5 + 3*6

    EXPECT_NEAR(nnMath::dot(v1, v2), expected, 1e-6);
}

TEST(MathTest, OuterProduct)
{
    std::vector<float> v1 = { 1.0f, 2.0f };
    std::vector<float> v2 = { 3.0f, 4.0f };

    auto result = nnMath::outer(v1, v2);

    std::vector<std::vector<float>> expected = {
        {3.0f, 4.0f},
        {6.0f, 8.0f}
    };

    EXPECT_EQ(result, expected);
}

}