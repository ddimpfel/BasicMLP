#pragma once
#include <vector>
#include <random>
#include <optional>
#include <functional>

#include "Layer.hpp"

class Network 
{
public:
    Network();
    Network(int layerCount,
        const std::function<float(float)> &ActivationFunction,
        const std::function<float(float)> &DerivativeActivationFunction,
        const std::function<float(size_t, float, float)> &LossFunction,
        const std::optional<std::vector<std::vector<std::vector<float>>>> &optWeights = std::nullopt,
        const std::optional<std::vector<std::vector<float>>> &optBiases = std::nullopt
    );

    std::vector<float> Predict(const std::vector<float>& inputs);

    void Fit(const std::vector<float> &inputs, const std::vector<float> &expectedOutputs);

    const std::vector<Layer>& GetLayers() const;
    std::vector<std::vector<std::vector<float>>> CopyWeights();

private:
    std::vector<Layer> m_layers;
    std::vector<std::vector<float>> m_layerOutputs;

    std::function<float(float)> pDerivativeActivationFunction; // TODO change to use 'reverse mode automatic differentiation'
    std::function<float(size_t, float, float)> pLossFunction;

    std::mt19937 m_gen;
    std::uniform_real_distribution<> m_dist;

    void InitRandomizer(int lower_bound, int upper_bound, std::optional<int> opt_seed = std::nullopt);

    void InitLayers(int layerCount,
        const std::function<float(float)> &ActivationFunction,
        const std::optional<std::vector<std::vector<std::vector<float>>>> &optWeights = std::nullopt,
        const std::optional<std::vector<std::vector<float>>> &optBiases = std::nullopt
    );

    /*!
     *   @brief Pass inputs forward through each layer
     *
     *      @param [in] inputs: std::vector<float>
     *
     *      @return m_layerOutputs.back() is prediction
     *      @note This function is O(layers * neurons * inputs)
     */
    std::vector<float>& ForwardPass(const std::vector<float>& inputs);

    /*!
     *  @brief Propagate the loss from output to input layer.
     *
     *      This is how the network learns. Loss is calculated at the outputs and utilized
     *      to update the weights. The partial derivative of the error with respect to the
     *      outputs gives the error a layer contributes to.  This updates the layers nodes
     *      and is also used to find the previous layers loss.
     *
     *      @param [in] expectedOutputs: std::vector<float>
     *      @param [in] learningRate: float
     */
    void BackwardPropagation(const std::vector<float>& expectedOutputs, float learningRate = 1);

    std::vector<float> CalculateLossPreviousLayer(std::vector<float> currentLoss, size_t currentLayer, size_t numNeurons);

    void ApplyGradients(
        std::vector<std::vector<std::vector<float>>>& weightGradients, 
        std::vector<std::vector<float>>& biasGradients, 
        float learningRate
    );
};