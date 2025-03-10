#pragma once

namespace param
{

const int WIDTH = 1280;
const int HEIGHT = 720;

int NoiseSize = 128;
int	NoiseOctaves = 4;
float fNoiseFreq = 0.01;
float fNoiseLacunarity = 2.f;
float fNoiseGain = 0.5f;

float fOffsetMultiplier = 100;
int VertexMultiplier = 20;

float fInnerRadiusMin = 100.f;
float fInnerRadiusScalar = 50.f;

float fPerlinPathWidth = 20.f;

}