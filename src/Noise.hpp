#pragma once
#include <random>

struct Vector2f
{
	float x;
	float y;

	Vector2f(float _x, float _y) :
		x(_x), y(_y) {}

	float dot(const Vector2f& other) const
	{
		return x * other.x + y * other.y;
	}
};

class PerlinNoiseGenerator
{
public:
	PerlinNoiseGenerator(int size);
	PerlinNoiseGenerator(int size, int seed);
	PerlinNoiseGenerator(int size, int seed, int octaves);
	~PerlinNoiseGenerator();

	float Noise(float x, float y) const;

	void setOctaves(int octaves);

private:
	void Permutate(std::vector<int>& permutations);

	const Vector2f& GetCornerConstantVector(int permutation) const;

	float Ease(float t) const;

	float Lerp(float a, float b, float t) const;

	int m_octaves = 4;
	std::vector<int> m_permutations;
	size_t m_size;

	std::vector<Vector2f> m_directions = {
		{ 1.f,  1.f},
		{-1.f,  1.f},
		{-1.f, -1.f},
		{ 1.f, -1.f}
	};

	std::random_device rd;
	std::mt19937 gen;
	std::uniform_real_distribution<float> dist{ 0.f, 1.f };
};