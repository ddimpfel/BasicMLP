#pragma once

#define PI 3.1415f

namespace param
{

// Window
const int iWIDTH = 1280;
const int iHEIGHT = 720;

// Noise
const int iSEED = 42;
int iNoiseSize = 128;
int	iNoiseOctaves = 4;
float fNoiseFreq = 0.01;
float fNoiseLacunarity = 2.f;
float fNoiseGain = 0.0f;
float fNoiseRadialMultiplier = 100;

// Path
int iVertexMultiplier = 10;
float fInnerRadiusMin = 212.f;
float fInnerRadiusScalar = 31.f;
float fPathWidth = 100.f;

// Box2d
const float fBOX2D_SCALE = 50.f;
enum CollisionCategories
{
    C_WALL = 0b0001,
    C_VEHICLE = 0b0010,
};

// Senses
size_t iRayCount = 3;
float fFieldOfView = PI;

// Agent
float collisionPenalizer = 0.1f;

}