#include "Noise.hpp"
#include <algorithm>
#include <cassert>

PerlinNoiseGenerator::PerlinNoiseGenerator(int size) 
{
	assert((size & (size - 1)) == 0, "Size must be a power of 2");
	gen.seed(rd());
	m_size = size;
	m_permutations.resize(m_size*2);
	Permutate(m_permutations);
}

PerlinNoiseGenerator::PerlinNoiseGenerator(int size, int seed)
{

}

PerlinNoiseGenerator::PerlinNoiseGenerator(int size, int seed, int octaves)
{

}

PerlinNoiseGenerator::~PerlinNoiseGenerator() = default;

float PerlinNoiseGenerator::Noise(float x, float y) const
{
	int x_int = static_cast<int>(x) & (m_size - 1); // same as % m_permutations.size()
	int y_int = static_cast<int>(y) & (m_size - 1); // since size is power of 2
	float x_frac = x - x_int;
	float y_frac = y - y_int;

	auto gradFromTopRight		= Vector2f(x_frac - 1.f, y_frac - 1.f);
	auto gradFromTopLeft		= Vector2f(x_frac,       y_frac - 1.f);
	auto gradFromBottomRight	= Vector2f(x_frac - 1.f, y_frac);
	auto gradFromBottomLeft		= Vector2f(x_frac	   , y_frac);

	auto& topRightConstantVector	= GetCornerConstantVector(m_permutations[m_permutations[x_int + 1] + y_int + 1]);
	auto& topLeftConstantVector		= GetCornerConstantVector(m_permutations[m_permutations[x_int]	   + y_int + 1]);
	auto& bottomRightConstantVector = GetCornerConstantVector(m_permutations[m_permutations[x_int + 1] + y_int]);
	auto& bottomLeftConstantVector  = GetCornerConstantVector(m_permutations[m_permutations[x_int]	   + y_int]);

	float dotTopRight	 = gradFromTopRight.dot(topRightConstantVector);
	float dotTopLeft	 = gradFromTopLeft.dot(topLeftConstantVector);
	float dotBottomLeft  = gradFromBottomRight.dot(bottomRightConstantVector);
	float dotBottomRight = gradFromBottomLeft.dot(bottomLeftConstantVector);

	float u = Ease(x_frac);
	float v = Ease(y_frac);

	return Lerp(u,
			Lerp(v, dotBottomLeft, dotTopLeft),
			Lerp(v, dotBottomRight, dotBottomLeft)
	);
}

void PerlinNoiseGenerator::setOctaves(int octaves) { m_octaves = octaves; }

void PerlinNoiseGenerator::Permutate(std::vector<int>& permutations)
{
	for (size_t i = 0; i < m_size; i++)
	{
		permutations[i] = i;
	}

	std::shuffle(permutations.begin(), permutations.end(), gen);

	for (size_t i = 0; i < m_size; i++)
	{
		permutations[i] = permutations[i];
	}
}

const Vector2f& PerlinNoiseGenerator::GetCornerConstantVector(int cornerIndex) const
{
	int dir = cornerIndex % m_directions.size();
	return m_directions[dir];
}

float PerlinNoiseGenerator::Ease(float t) const
{
    return ((6*t - 15)*t + 10) * t*t*t;
}

float PerlinNoiseGenerator::Lerp(float a, float b, float t) const
{
	return a + (b - a)*t;
}
