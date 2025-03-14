#pragma once

namespace param
{

// Window
const int WIDTH = 1280;
const int HEIGHT = 720;

// Noise
const int SEED = 42;
int NoiseSize = 128;
int	NoiseOctaves = 4;
float fNoiseFreq = 0.01;
float fNoiseLacunarity = 2.f;
float fNoiseGain = 0.0f;
float fNoiseRadialMultiplier = 100;

// Path
int VertexMultiplier = 10;
float fInnerRadiusMin = 212.f;
float fInnerRadiusScalar = 31.f;
float fPathWidth = 75.f;

// Box2d
const float fBOX2D_SCALE = 50.f;

}