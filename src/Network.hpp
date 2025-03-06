#pragma once
#include <vector>
#include <random>
#include <memory>
#include <optional>
#include <functional>

#include "Layer.hpp"

class Network 
{
public:
    Network();
    Network(
        const std::vector<int>& initialLayers,
        const std::function<float(float)>& ActivationFunction,
        const std::function<float(float)>& DerivativeActivationFunction,
        const std::function<float(size_t, float, float)>& LossFunction,
        std::unique_ptr<std::uniform_real_distribution<float>> dist = nullptr,
        std::unique_ptr<std::mt19937> gen = nullptr
    );
    Network(
        const std::vector<int>& initialLayers,
        const std::function<float(float)>& ActivationFunction,
        const std::function<float(float)>& DerivativeActivationFunction,
        const std::function<float(size_t, float, float)>& LossFunction,
        std::vector<std::vector<std::vector<float>>>& optWeights,
        std::vector<std::vector<float>>& optBiases
    );

    std::vector<float> Predict(const std::vector<float>& inputs);

    void Fit(const std::vector<float> &inputs, const std::vector<float> &expectedOutputs);

    const std::vector<int>& getArchitecture() const;

    const std::vector<Layer>& getLayers() const;

    const std::vector<std::vector<float>>& getLayerOutputs() const;

    void setLearningRate(float lr)
    {
        m_learningRate = lr;
    }

    /*!
     *  @return hard copy of weights 3d vector
     */
    std::vector<std::vector<std::vector<float>>> copyWeights();

    /*!
     *  @return hard copy of biases 2d vector
     */
    std::vector<std::vector<float>> copyBiases();

private:
    void InitLayersPreset(
        const std::function<float(float)> &ActivationFunction,
        const std::optional<std::vector<std::vector<std::vector<float>>>> &optWeights = std::nullopt,
        const std::optional<std::vector<std::vector<float>>> &optBiases = std::nullopt
    );

    void InitLayersRandom(
        const std::function<float(float)>& ActivationFunction,
        std::uniform_real_distribution<float> dist,
        std::mt19937 gen
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
    void BackwardPropagation(const std::vector<float>& expectedOutputs);

    std::vector<float> CalculateLossPreviousLayer(std::vector<float> currentLoss, size_t currentLayer, size_t numNeurons);

    void ApplyGradients(
        std::vector<std::vector<std::vector<float>>>& weightGradients, 
        std::vector<std::vector<float>>& biasGradients
    );

    std::vector<int> m_architecture;
    std::vector<Layer> m_layers;
    std::vector<std::vector<float>> m_layerOutputs;

    float m_learningRate;
    std::function<float(float)> DerivativeActivationFunction; // TODO change to use 'reverse mode automatic differentiation'
    std::function<float(size_t, float, float)> LossFunction;
};