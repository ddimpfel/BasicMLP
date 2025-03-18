#pragma once
#include <vector>
#include <crtdbg.h>
#include <iomanip>
#include <ios>
#include <iosfwd>
#include <iostream>
#include <sstream>
#include <string>

namespace nnMath
{

// Element-wise multiplication
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

// Dot product
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

// Outer product returns matrix with u rows by v columns
static std::vector<std::vector<float>> outer(std::vector<float> u, std::vector<float> v)
{
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

// Sum a single vector
static float sum(const std::vector<float>& vec)
{
    float sum = 0;
    for (const auto &el : vec)
    {
        sum += el;
    }
    return sum;
}

// Print a NxM matrix to console and give it some name as a label
static void printMat(const std::vector<std::vector<float>>& matrix, std::string name)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(8);

    oss << name << " [";
    for (size_t i = 0; i < matrix.size(); ++i) {
        oss << "[";
        for (size_t j = 0; j < matrix[i].size(); ++j) {
            oss << matrix[i][j];
            if (j < matrix[i].size() - 1) {
                oss << " ";
            }
        }
        oss << "]";
        if (i < matrix.size() - 1) {
            oss << " ";
        }
    }
    oss << "]";

    std::cout << oss.str() << std::endl;
}

// Print a vector to console and give it some name as a label
static void printVec(const std::vector<float>& vector, std::string name)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(8);

    oss << name << " [";
    for (size_t i = 0; i < vector.size(); ++i) {
        oss << vector[i];
        if (i < vector.size() - 1) {
            oss << " ";
        }
    }
    oss << "]";

    std::cout << oss.str() << std::endl;
}

} // namespace nnMath
