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
float fPathWidth = 150.f;

// Box2d
const float fBOX2D_SCALE = 50.f;
enum CollisionCategories
{
    C_WALL = 0b0001,
    C_VEHICLE = 0b0010,
};

// Agent
// Size
float fHalfWidth = 0.3f; // meters
float fHalfHeight = 0.15f; // meters
float fStartX = -280.f / fBOX2D_SCALE;
float fStartY = 0.f;
float fStartRotation = -PI / 2.f;

// Senses
size_t uRayCount = 5;
float fFieldOfView = PI / 1.5f;

// Simulation
size_t uAgentCount = 50;
float fCollisionPenalizer = 10.f;
float fDistanceMultiplier = 35.f;
int uSimulationSpeed = 3;
float fGenerationTimer = 60.f;
float fBestPerformersFraction = 0.2f;

}