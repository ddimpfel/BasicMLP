#pragma once
#include <vector>
#include <crtdbg.h>
#include <iomanip>
#include <ios>
#include <iosfwd>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>

namespace nnMath
{

static std::vector<float> mult(std::vector<float> lhs, std::vector<float> rhs) 
{
    _ASSERT(lhs.size() == rhs.size());

    std::vector<float> vecOut(lhs.size());
    for (size_t i = 0; i < lhs.size(); i++)
    {
        vecOut[i] = lhs[i] * rhs[i];
    }

    return vecOut;
}

static float dot(std::vector<float> lhs, std::vector<float> rhs)
{
    _ASSERT(lhs.size() == rhs.size());

    float fProduct = 0;
    for (size_t i = 0; i < lhs.size(); i++)
    {
        fProduct += lhs[i] * rhs[i];
    }

    return fProduct;
}

static std::vector<std::vector<float>> outer(std::vector<float> u, std::vector<float> v)
{
    _ASSERT(u.size() == v.size());

    std::vector<std::vector<float>> matOut(u.size(), std::vector<float>(v.size()));
    for (size_t r = 0; r < u.size(); r++)
    {
        for (size_t c = 0; c < v.size(); c++)
        {
            matOut[r][c] = u[r] * v[c];
        }
    }

    return matOut;
}

static float sum(const std::vector<float>& vec)
{
    float sum = 0;
    for (const auto &el : vec)
    {
        sum += el;
    }
    return sum;
}

static void printMat(const std::vector<std::vector<float>>& weightGradients, std::string name)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(8);

    oss << name << " [";
    for (size_t i = 0; i < weightGradients.size(); ++i) {
        oss << "[";
        for (size_t j = 0; j < weightGradients[i].size(); ++j) {
            oss << weightGradients[i][j];
            if (j < weightGradients[i].size() - 1) {
                oss << " ";
            }
        }
        oss << "]";
        if (i < weightGradients.size() - 1) {
            oss << " ";
        }
    }
    oss << "]";

    std::cout << oss.str() << std::endl;
}

static void printVec(const std::vector<float>& delta, std::string name)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(8);

    oss << name << " [";
    for (size_t i = 0; i < delta.size(); ++i) {
        oss << delta[i];
        if (i < delta.size() - 1) {
            oss << " ";
        }
    }
    oss << "]";

    std::cout << oss.str() << std::endl;
}

} // namespace nnMath
