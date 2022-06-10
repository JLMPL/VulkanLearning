/**********************************************************************************************
*
*   raymath v1.2 - Math functions to work with vec3, mat4 and Quaternions
*
*   CONFIGURATION:
*
*   #define RAYMATH_IMPLEMENTATION
*       Generates the implementation of the library into the included file.
*       If not defined, the library is in header only mode and can be included in other headers
*       or source files without problems. But only ONE file should hold the implementation.
*
*   #define RAYMATH_HEADER_ONLY
*       Define static inline functions code, so #include header suffices for use.
*       This may use up lots of memory.
*
*   #define RAYMATH_STANDALONE
*       Avoid raylib.h header inclusion in this file.
*       vec3 and mat4 data types are defined internally in raymath module.
*
*
*   LICENSE: zlib/libpng
*
*   Copyright (c) 2015-2019 Ramon Santamaria (@raysan5)
*
*   This software is provided "as-is", without any express or implied warranty. In no event
*   will the authors be held liable for any damages arising from the use of this software.
*
*   Permission is granted to anyone to use this software for any purpose, including commercial
*   applications, and to alter it and redistribute it freely, subject to the following restrictions:
*
*     1. The origin of this software must not be misrepresented; you must not claim that you
*     wrote the original software. If you use this software in a product, an acknowledgment
*     in the product documentation would be appreciated but is not required.
*
*     2. Altered source versions must be plainly marked as such, and must not be misrepresented
*     as being the original software.
*
*     3. This notice may not be removed or altered from any source distribution.
*
**********************************************************************************************/

/**********************************************************************************************
*   This is a modified version of the original raymath.h
*   Copywrong (C) 2021-2022 Me (@Hoboker)
**********************************************************************************************/

#ifndef RAYMATH_H
#define RAYMATH_H

#define RAYMATH_STANDALONE      // NOTE: To use raymath as standalone lib, just uncomment this line
#define RAYMATH_HEADER_ONLY     // NOTE: To compile functions as static inline, uncomment this line

#ifndef RAYMATH_STANDALONE
    #include "raylib.h"           // Required for structs: vec3, mat4
#endif

#if defined(RAYMATH_IMPLEMENTATION) && defined(RAYMATH_HEADER_ONLY)
    #error "Specifying both RAYMATH_IMPLEMENTATION and RAYMATH_HEADER_ONLY is contradictory"
#endif

#if defined(RAYMATH_IMPLEMENTATION)
    #if defined(_WIN32) && defined(BUILD_LIBTYPE_SHARED)
        #define RMDEF __declspec(dllexport) extern inline // We are building raylib as a Win32 shared library (.dll).
    #elif defined(_WIN32) && defined(USE_LIBTYPE_SHARED)
        #define RMDEF __declspec(dllimport)         // We are using raylib as a Win32 shared library (.dll)
    #else
        #define RMDEF extern inline // Provide external definition
    #endif
#elif defined(RAYMATH_HEADER_ONLY)
    #define RMDEF static inline // Functions may be inlined, no external out-of-line definition
#else
    #if defined(__TINYC__)
        #define RMDEF static inline // plain inline not supported by tinycc (See issue #435)
    #else
        #define RMDEF inline        // Functions may be inlined or external definition used
    #endif
#endif

//----------------------------------------------------------------------------------
// Defines and Macros
//----------------------------------------------------------------------------------
#ifndef PI
    #define PI 3.14159265358979323846f
#endif

#ifndef DEG2RAD
    #define DEG2RAD (PI/180.0f)
#endif

#ifndef RAD2DEG
    #define RAD2DEG (180.0f/PI)
#endif

// Return float vector for mat4
#ifndef MatrixToFloat
    #define MatrixToFloat(mat) (MatrixToFloatV(mat).v)
#endif

// Return float vector for vec3
#ifndef Vector3ToFloat
    #define Vector3ToFloat(vec) (Vector3ToFloatV(vec).v)
#endif

//----------------------------------------------------------------------------------
// Types and Structures Definition
//----------------------------------------------------------------------------------

#if defined(RAYMATH_STANDALONE)
    // vec2 type
    typedef struct vec2 {
        float x;
        float y;
    } vec2;

    // vec3 type
    typedef struct vec3 {
        float x;
        float y;
        float z;
    } vec3;

    typedef struct vec4 {
        float x;
        float y;
        float z;
        float w;
    } vec4;

    // quat type
    typedef struct quat {
        float x;
        float y;
        float z;
        float w;
    } quat;

    // mat4 type (OpenGL style 4x4 - right handed, row major)
/*    typedef struct mat4 {
        float m0, m4, m8, m12;
        float m1, m5, m9, m13;
        float m2, m6, m10, m14;
        float m3, m7, m11, m15;
    } mat4;*/

    // mat4 type (OpenGL style 4x4 - right handed, column major)
    typedef struct mat4 {
		union
		{
			struct
			{
				float m0, m1, m2, m3;
				float m4, m5, m6, m7;
				float m8, m9, m10, m11;
				float m12, m13, m14, m15;
			};

			vec4 v[4];
		};
    } mat4;
#endif

// NOTE: Helper types to be used instead of array return types for *ToFloat functions
typedef struct float3 { float v[3]; } float3;
typedef struct float16 { float v[16]; } float16;

#include <math.h>       // Required for: sinf(), cosf(), tan(), fabs()
#include <string.h>     // memset
#include <stdio.h>

//my annoyance is over

#define v3 Vector3 //remove this
#define v3_add Vector3Add
#define v3_sub Vector3Subtract
#define v3_mul Vector3Multiply
#define v3_div Vector3Divide
#define v3_tr Vector3Transform
#define v3_dot Vector3DotProduct
#define v3_cross Vector3CrossProduct
#define v3_norm Vector3Normalize
#define v3_dist Vector3Distance
#define v3_len Vector3Length

#define v2_add Vector2Add

#define m4_id MatrixIdentity
#define m4_mul MatrixMultiply
#define m4_inv MatrixInvert
#define m4_scale MatrixScale
#define m4_transpose MatrixTranspose
#define m4_translate MatrixTranslate
#define m4_rotate MatrixRotate
#define m4_rotate_y MatrixRotateY
#define m4_rotate_x MatrixRotateX
#define m4_add MatrixAdd

#define Vec2 Vector2
#define Vec3 Vector3

//----------------------------------------------------------------------------------
// Module Functions Definition - Utils math
//----------------------------------------------------------------------------------

// Clamp float value
RMDEF float Clamp(float value, float min, float max)
{
    const float res = value < min ? min : value;
    return res > max ? max : res;
}

// Calculate linear interpolation between two floats
RMDEF float Lerp(float start, float end, float amount)
{
    return start + amount*(end - start);
}

//----------------------------------------------------------------------------------
// Module Functions Definition - vec2 math
//----------------------------------------------------------------------------------

RMDEF vec2 Vector2(float x, float y)
{
    vec2 result = {x,y};
    return result;
}

RMDEF vec2 vec2_2i(int x, int y)
{
    return Vector2((float)x, (float)y);
}

// Vector with components value 0.0f
RMDEF vec2 Vector2Zero(void)
{
    vec2 result = { 0.0f, 0.0f };
    return result;
}

// Vector with components value 1.0f
RMDEF vec2 Vector2One(void)
{
    vec2 result = { 1.0f, 1.0f };
    return result;
}

// Add two vectors (v1 + v2)
RMDEF vec2 Vector2Add(vec2 v1, vec2 v2)
{
    vec2 result = { v1.x + v2.x, v1.y + v2.y };
    return result;
}

// Subtract two vectors (v1 - v2)
RMDEF vec2 Vector2Subtract(vec2 v1, vec2 v2)
{
    vec2 result = { v1.x - v2.x, v1.y - v2.y };
    return result;
}

// Calculate vector length
RMDEF float Vector2Length(vec2 v)
{
    float result = sqrtf((v.x*v.x) + (v.y*v.y));
    return result;
}

// Calculate two vectors dot product
RMDEF float Vector2DotProduct(vec2 v1, vec2 v2)
{
    float result = (v1.x*v2.x + v1.y*v2.y);
    return result;
}

// Calculate distance between two vectors
RMDEF float Vector2Distance(vec2 v1, vec2 v2)
{
    float result = sqrtf((v1.x - v2.x)*(v1.x - v2.x) + (v1.y - v2.y)*(v1.y - v2.y));
    return result;
}

// Calculate angle from two vectors in X-axis
RMDEF float Vector2Angle(vec2 v1, vec2 v2)
{
    float result = atan2f(v2.y - v1.y, v2.x - v1.x)*(180.0f/(float)PI);
    if (result < 0) result += 360.0f;
    return result;
}

// Scale vector (multiply by value)
RMDEF vec2 Vector2Scale(vec2 v, float scale)
{
    vec2 result = { v.x*scale, v.y*scale };
    return result;
}

// Multiply vector by vector
RMDEF vec2 Vector2MultiplyV(vec2 v1, vec2 v2)
{
    vec2 result = { v1.x*v2.x, v1.y*v2.y };
    return result;
}

RMDEF vec2 Vector2Multiply(vec2 v, float s)
{
    vec2 result = { v.x*s, v.y*s };
    return result;
}

// Negate vector
RMDEF vec2 Vector2Negate(vec2 v)
{
    vec2 result = { -v.x, -v.y };
    return result;
}

// Divide vector by a float value
RMDEF vec2 Vector2Divide(vec2 v, float div)
{
    vec2 result = { v.x/div, v.y/div };
    return result;
}

// Divide vector by vector
RMDEF vec2 Vector2DivideV(vec2 v1, vec2 v2)
{
    vec2 result = { v1.x/v2.x, v1.y/v2.y };
    return result;
}

// Normalize provided vector
RMDEF vec2 Vector2Normalize(vec2 v)
{
    vec2 result = Vector2Divide(v, Vector2Length(v));
    return result;
}

// Calculate linear interpolation between two vectors
RMDEF vec2 Vector2Lerp(vec2 v1, vec2 v2, float amount)
{
    vec2 result = { 0 };

    result.x = v1.x + amount*(v2.x - v1.x);
    result.y = v1.y + amount*(v2.y - v1.y);

    return result;
}

//----------------------------------------------------------------------------------
// Module Functions Definition - vec3 math
//----------------------------------------------------------------------------------

RMDEF vec3 Vector3(float x, float y, float z)
{
    //return (vec3){x,y,z};

    vec3 vec;
    vec.x = x;
    vec.y = y;
    vec.z = z;
    return vec;
}

// Vector with components value 0.0f
RMDEF vec3 Vector3Zero(void)
{
    vec3 result = { 0.0f, 0.0f, 0.0f };
    return result;
}

// Vector with components value 1.0f
RMDEF vec3 Vector3One(void)
{
    vec3 result = { 1.0f, 1.0f, 1.0f };
    return result;
}

// Add two vectors
RMDEF vec3 Vector3Add(vec3 v1, vec3 v2)
{
    vec3 result = { v1.x + v2.x, v1.y + v2.y, v1.z + v2.z };
    return result;
}

// Subtract two vectors
RMDEF vec3 Vector3Subtract(vec3 v1, vec3 v2)
{
    vec3 result = { v1.x - v2.x, v1.y - v2.y, v1.z - v2.z };
    return result;
}

// Multiply vector by scalar
RMDEF vec3 Vector3Multiply(vec3 v, float scalar)
{
    vec3 result = { v.x*scalar, v.y*scalar, v.z*scalar };
    return result;
}

// Multiply vector by vector
RMDEF vec3 Vector3MultiplyV(vec3 v1, vec3 v2)
{
    vec3 result = { v1.x*v2.x, v1.y*v2.y, v1.z*v2.z };
    return result;
}

// Calculate two vectors cross product
RMDEF vec3 Vector3CrossProduct(vec3 v1, vec3 v2)
{
    vec3 result = { v1.y*v2.z - v1.z*v2.y, v1.z*v2.x - v1.x*v2.z, v1.x*v2.y - v1.y*v2.x };
    return result;
}

// Calculate one vector perpendicular vector
RMDEF vec3 Vector3Perpendicular(vec3 v)
{
    vec3 result = { 0 };

    float min = (float) fabs(v.x);
    vec3 cardinalAxis = {1.0f, 0.0f, 0.0f};

    if (fabs(v.y) < min)
    {
        min = (float) fabs(v.y);
        vec3 tmp = {0.0f, 1.0f, 0.0f};
        cardinalAxis = tmp;
    }

if (fabs(v.z) < min)
    {
        vec3 tmp = {0.0f, 0.0f, 1.0f};
        cardinalAxis = tmp;
    }

    result = Vector3CrossProduct(v, cardinalAxis);

    return result;
}

// Calculate vector length
RMDEF float Vector3Length(const vec3 v)
{
    float result = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
    return result;
}

// Calculate squared vector length
RMDEF float Vector3Length2(const vec3 v)
{
    return v.x*v.x + v.y*v.y + v.z*v.z;
}

// Calculate two vectors dot product
RMDEF float Vector3DotProduct(vec3 v1, vec3 v2)
{
    float result = (v1.x*v2.x + v1.y*v2.y + v1.z*v2.z);
    return result;
}

// Calculate distance between two vectors
RMDEF float Vector3Distance(vec3 v1, vec3 v2)
{
    float dx = v2.x - v1.x;
    float dy = v2.y - v1.y;
    float dz = v2.z - v1.z;
    float result = sqrtf(dx*dx + dy*dy + dz*dz);
    return result;
}

// Calculate squared distance between two vectors
RMDEF float Vector3Distance2(vec3 v1, vec3 v2)
{
    float dx = v2.x - v1.x;
    float dy = v2.y - v1.y;
    float dz = v2.z - v1.z;
    return dx*dx + dy*dy + dz*dz;
}

// Scale provided vector
RMDEF vec3 Vector3Scale(vec3 v, float scale)
{
    vec3 result = { v.x*scale, v.y*scale, v.z*scale };
    return result;
}

// Negate provided vector (invert direction)
RMDEF vec3 Vector3Negate(vec3 v)
{
    vec3 result = { -v.x, -v.y, -v.z };
    return result;
}

// Divide vector by a float value
RMDEF vec3 Vector3Divide(vec3 v, float div)
{
    vec3 result = { v.x / div, v.y / div, v.z / div };
    return result;
}

// Divide vector by vector
RMDEF vec3 Vector3DivideV(vec3 v1, vec3 v2)
{
    vec3 result = { v1.x/v2.x, v1.y/v2.y, v1.z/v2.z };
    return result;
}

// Normalize provided vector
RMDEF vec3 Vector3Normalize(vec3 v)
{
    vec3 result = v;

    float length, ilength;
    length = Vector3Length(v);
    if (length == 0.0f) length = 1.0f;
    ilength = 1.0f/length;

    result.x *= ilength;
    result.y *= ilength;
    result.z *= ilength;

    return result;
}

// Orthonormalize provided vectors
// Makes vectors normalized and orthogonal to each other
// Gram-Schmidt function implementation
RMDEF void Vector3OrthoNormalize(vec3 *v1, vec3 *v2)
{
    *v1 = Vector3Normalize(*v1);
    vec3 vn = Vector3CrossProduct(*v1, *v2);
    vn = Vector3Normalize(vn);
    *v2 = Vector3CrossProduct(vn, *v1);
}

// Transforms a vec3 by a given mat4
RMDEF vec3 Vector3Transform(vec3 v, mat4 mat)
{
    vec3 result = { 0 };
    float x = v.x;
    float y = v.y;
    float z = v.z;

    result.x = mat.m0*x + mat.m4*y + mat.m8*z + mat.m12;
    result.y = mat.m1*x + mat.m5*y + mat.m9*z + mat.m13;
    result.z = mat.m2*x + mat.m6*y + mat.m10*z + mat.m14;

    return result;
}

// Transform a vector by quaternion rotation
RMDEF vec3 Vector3RotateByQuaternion(vec3 v, quat q)
{
    vec3 result = { 0 };

    result.x = v.x*(q.x*q.x + q.w*q.w - q.y*q.y - q.z*q.z) + v.y*(2*q.x*q.y - 2*q.w*q.z) + v.z*(2*q.x*q.z + 2*q.w*q.y);
    result.y = v.x*(2*q.w*q.z + 2*q.x*q.y) + v.y*(q.w*q.w - q.x*q.x + q.y*q.y - q.z*q.z) + v.z*(-2*q.w*q.x + 2*q.y*q.z);
    result.z = v.x*(-2*q.w*q.y + 2*q.x*q.z) + v.y*(2*q.w*q.x + 2*q.y*q.z)+ v.z*(q.w*q.w - q.x*q.x - q.y*q.y + q.z*q.z);

    return result;
}

// Calculate linear interpolation between two vectors
RMDEF vec3 Vector3Lerp(vec3 v1, vec3 v2, float amount)
{
    vec3 result = { 0 };

    result.x = v1.x + amount*(v2.x - v1.x);
    result.y = v1.y + amount*(v2.y - v1.y);
    result.z = v1.z + amount*(v2.z - v1.z);

    return result;
}

// Calculate reflected vector to normal
RMDEF vec3 Vector3Reflect(vec3 v, vec3 normal)
{
    // I is the original vector
    // N is the normal of the incident plane
    // R = I - (2*N*( DotProduct[ I,N] ))

    vec3 result = { 0 };

    float dotProduct = Vector3DotProduct(v, normal);

    result.x = v.x - (2.0f*normal.x)*dotProduct;
    result.y = v.y - (2.0f*normal.y)*dotProduct;
    result.z = v.z - (2.0f*normal.z)*dotProduct;

    return result;
}

// Return min value for each pair of components
RMDEF vec3 Vector3Min(vec3 v1, vec3 v2)
{
    vec3 result = { 0 };

    result.x = fminf(v1.x, v2.x);
    result.y = fminf(v1.y, v2.y);
    result.z = fminf(v1.z, v2.z);

    return result;
}

// Return max value for each pair of components
RMDEF vec3 Vector3Max(vec3 v1, vec3 v2)
{
    vec3 result = { 0 };

    result.x = fmaxf(v1.x, v2.x);
    result.y = fmaxf(v1.y, v2.y);
    result.z = fmaxf(v1.z, v2.z);

    return result;
}

// Compute barycenter coordinates (u, v, w) for point p with respect to triangle (a, b, c)
// NOTE: Assumes P is on the plane of the triangle
RMDEF vec3 Vector3Barycenter(vec3 p, vec3 a, vec3 b, vec3 c)
{
    //Vector v0 = b - a, v1 = c - a, v2 = p - a;

    vec3 v0 = Vector3Subtract(b, a);
    vec3 v1 = Vector3Subtract(c, a);
    vec3 v2 = Vector3Subtract(p, a);
    float d00 = Vector3DotProduct(v0, v0);
    float d01 = Vector3DotProduct(v0, v1);
    float d11 = Vector3DotProduct(v1, v1);
    float d20 = Vector3DotProduct(v2, v0);
    float d21 = Vector3DotProduct(v2, v1);

    float denom = d00*d11 - d01*d01;

    vec3 result = { 0 };

    result.y = (d11*d20 - d01*d21)/denom;
    result.z = (d00*d21 - d01*d20)/denom;
    result.x = 1.0f - (result.z + result.y);

    return result;
}

// Returns vec3 as float array
RMDEF float3 Vector3ToFloatV(vec3 v)
{
    float3 buffer = { 0 };

    buffer.v[0] = v.x;
    buffer.v[1] = v.y;
    buffer.v[2] = v.z;

    return buffer;
}

//----------------------------------------------------------------------------------
// Module Functions Definition - mat4 math
//----------------------------------------------------------------------------------

// Compute matrix determinant
RMDEF float MatrixDeterminant(mat4 mat)
{
    float result = { 0 };

    // Cache the matrix values (speed optimization)
    float a00 = mat.m0, a01 = mat.m1, a02 = mat.m2, a03 = mat.m3;
    float a10 = mat.m4, a11 = mat.m5, a12 = mat.m6, a13 = mat.m7;
    float a20 = mat.m8, a21 = mat.m9, a22 = mat.m10, a23 = mat.m11;
    float a30 = mat.m12, a31 = mat.m13, a32 = mat.m14, a33 = mat.m15;

    result = a30*a21*a12*a03 - a20*a31*a12*a03 - a30*a11*a22*a03 + a10*a31*a22*a03 +
             a20*a11*a32*a03 - a10*a21*a32*a03 - a30*a21*a02*a13 + a20*a31*a02*a13 +
             a30*a01*a22*a13 - a00*a31*a22*a13 - a20*a01*a32*a13 + a00*a21*a32*a13 +
             a30*a11*a02*a23 - a10*a31*a02*a23 - a30*a01*a12*a23 + a00*a31*a12*a23 +
             a10*a01*a32*a23 - a00*a11*a32*a23 - a20*a11*a02*a33 + a10*a21*a02*a33 +
             a20*a01*a12*a33 - a00*a21*a12*a33 - a10*a01*a22*a33 + a00*a11*a22*a33;

    return result;
}

// Returns the trace of the matrix (sum of the values along the diagonal)
RMDEF float MatrixTrace(mat4 mat)
{
    float result = (mat.m0 + mat.m5 + mat.m10 + mat.m15);
    return result;
}

// Transposes provided matrix
RMDEF mat4 MatrixTranspose(mat4 mat)
{
    mat4 result = { 0 };

    result.m0 = mat.m0;
    result.m1 = mat.m4;
    result.m2 = mat.m8;
    result.m3 = mat.m12;
    result.m4 = mat.m1;
    result.m5 = mat.m5;
    result.m6 = mat.m9;
    result.m7 = mat.m13;
    result.m8 = mat.m2;
    result.m9 = mat.m6;
    result.m10 = mat.m10;
    result.m11 = mat.m14;
    result.m12 = mat.m3;
    result.m13 = mat.m7;
    result.m14 = mat.m11;
    result.m15 = mat.m15;

    return result;
}

// Invert provided matrix
RMDEF mat4 MatrixInvert(mat4 mat)
{
    mat4 result = { 0 };

    // Cache the matrix values (speed optimization)
    float a00 = mat.m0, a01 = mat.m1, a02 = mat.m2, a03 = mat.m3;
    float a10 = mat.m4, a11 = mat.m5, a12 = mat.m6, a13 = mat.m7;
    float a20 = mat.m8, a21 = mat.m9, a22 = mat.m10, a23 = mat.m11;
    float a30 = mat.m12, a31 = mat.m13, a32 = mat.m14, a33 = mat.m15;

    float b00 = a00*a11 - a01*a10;
    float b01 = a00*a12 - a02*a10;
    float b02 = a00*a13 - a03*a10;
    float b03 = a01*a12 - a02*a11;
    float b04 = a01*a13 - a03*a11;
    float b05 = a02*a13 - a03*a12;
    float b06 = a20*a31 - a21*a30;
    float b07 = a20*a32 - a22*a30;
    float b08 = a20*a33 - a23*a30;
    float b09 = a21*a32 - a22*a31;
    float b10 = a21*a33 - a23*a31;
    float b11 = a22*a33 - a23*a32;

    // Calculate the invert determinant (inlined to avoid double-caching)
    float invDet = 1.0f/(b00*b11 - b01*b10 + b02*b09 + b03*b08 - b04*b07 + b05*b06);

    result.m0 = (a11*b11 - a12*b10 + a13*b09)*invDet;
    result.m1 = (-a01*b11 + a02*b10 - a03*b09)*invDet;
    result.m2 = (a31*b05 - a32*b04 + a33*b03)*invDet;
    result.m3 = (-a21*b05 + a22*b04 - a23*b03)*invDet;
    result.m4 = (-a10*b11 + a12*b08 - a13*b07)*invDet;
    result.m5 = (a00*b11 - a02*b08 + a03*b07)*invDet;
    result.m6 = (-a30*b05 + a32*b02 - a33*b01)*invDet;
    result.m7 = (a20*b05 - a22*b02 + a23*b01)*invDet;
    result.m8 = (a10*b10 - a11*b08 + a13*b06)*invDet;
    result.m9 = (-a00*b10 + a01*b08 - a03*b06)*invDet;
    result.m10 = (a30*b04 - a31*b02 + a33*b00)*invDet;
    result.m11 = (-a20*b04 + a21*b02 - a23*b00)*invDet;
    result.m12 = (-a10*b09 + a11*b07 - a12*b06)*invDet;
    result.m13 = (a00*b09 - a01*b07 + a02*b06)*invDet;
    result.m14 = (-a30*b03 + a31*b01 - a32*b00)*invDet;
    result.m15 = (a20*b03 - a21*b01 + a22*b00)*invDet;

    return result;
}

// Normalize provided matrix
RMDEF mat4 MatrixNormalize(mat4 mat)
{
    mat4 result = { 0 };

    float det = MatrixDeterminant(mat);

    result.m0 = mat.m0/det;
    result.m1 = mat.m1/det;
    result.m2 = mat.m2/det;
    result.m3 = mat.m3/det;
    result.m4 = mat.m4/det;
    result.m5 = mat.m5/det;
    result.m6 = mat.m6/det;
    result.m7 = mat.m7/det;
    result.m8 = mat.m8/det;
    result.m9 = mat.m9/det;
    result.m10 = mat.m10/det;
    result.m11 = mat.m11/det;
    result.m12 = mat.m12/det;
    result.m13 = mat.m13/det;
    result.m14 = mat.m14/det;
    result.m15 = mat.m15/det;

    return result;
}

// Returns identity matrix
RMDEF mat4 MatrixIdentity(void)
{
/*    mat4 result = { 1.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 1.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 1.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 1.0f };*/

    mat4 result;
    memset(&result, 0, sizeof(mat4));

    result.m0 = 1.f;
    result.m5 = 1.f;
    result.m10 = 1.f;
    result.m15 = 1.f;

    return result;
}

// Add two matrices
RMDEF mat4 MatrixAdd(mat4 left, mat4 right)
{
    mat4 result = MatrixIdentity();

    result.m0 = left.m0 + right.m0;
    result.m1 = left.m1 + right.m1;
    result.m2 = left.m2 + right.m2;
    result.m3 = left.m3 + right.m3;
    result.m4 = left.m4 + right.m4;
    result.m5 = left.m5 + right.m5;
    result.m6 = left.m6 + right.m6;
    result.m7 = left.m7 + right.m7;
    result.m8 = left.m8 + right.m8;
    result.m9 = left.m9 + right.m9;
    result.m10 = left.m10 + right.m10;
    result.m11 = left.m11 + right.m11;
    result.m12 = left.m12 + right.m12;
    result.m13 = left.m13 + right.m13;
    result.m14 = left.m14 + right.m14;
    result.m15 = left.m15 + right.m15;

    return result;
}

// Subtract two matrices (left - right)
RMDEF mat4 MatrixSubtract(mat4 left, mat4 right)
{
    mat4 result = MatrixIdentity();

    result.m0 = left.m0 - right.m0;
    result.m1 = left.m1 - right.m1;
    result.m2 = left.m2 - right.m2;
    result.m3 = left.m3 - right.m3;
    result.m4 = left.m4 - right.m4;
    result.m5 = left.m5 - right.m5;
    result.m6 = left.m6 - right.m6;
    result.m7 = left.m7 - right.m7;
    result.m8 = left.m8 - right.m8;
    result.m9 = left.m9 - right.m9;
    result.m10 = left.m10 - right.m10;
    result.m11 = left.m11 - right.m11;
    result.m12 = left.m12 - right.m12;
    result.m13 = left.m13 - right.m13;
    result.m14 = left.m14 - right.m14;
    result.m15 = left.m15 - right.m15;

    return result;
}

// Returns translation matrix
RMDEF mat4 MatrixTranslate(float x, float y, float z)
{
/*    mat4 result = { 1.0f, 0.0f, 0.0f, x,
                      0.0f, 1.0f, 0.0f, y,
                      0.0f, 0.0f, 1.0f, z,
                      0.0f, 0.0f, 0.0f, 1.0f };*/

    mat4 result = MatrixIdentity();

    result.m12 = x;
    result.m13 = y;
    result.m14 = z;

    return result;
}

// not that ^
RMDEF vec3 MatrixTranslation(mat4 tr)
{
    vec3 result;
    result.x = tr.m12;
    result.y = tr.m13;
    result.z = tr.m14;
    return result;
}

// Create rotation matrix from axis and angle
// NOTE: Angle should be provided in radians
RMDEF mat4 MatrixRotate(vec3 axis, float angle)
{
    mat4 result = { 0 };

    float x = axis.x, y = axis.y, z = axis.z;

    float length = sqrtf(x*x + y*y + z*z);

    if ((length != 1.0f) && (length != 0.0f))
    {
        length = 1.0f/length;
        x *= length;
        y *= length;
        z *= length;
    }

    float sinres = sinf(angle);
    float cosres = cosf(angle);
    float t = 1.0f - cosres;

    result.m0  = x*x*t + cosres;
    result.m1  = y*x*t + z*sinres;
    result.m2  = z*x*t - y*sinres;
    result.m3  = 0.0f;

    result.m4  = x*y*t - z*sinres;
    result.m5  = y*y*t + cosres;
    result.m6  = z*y*t + x*sinres;
    result.m7  = 0.0f;

    result.m8  = x*z*t + y*sinres;
    result.m9  = y*z*t - x*sinres;
    result.m10 = z*z*t + cosres;
    result.m11 = 0.0f;

    result.m12 = 0.0f;
    result.m13 = 0.0f;
    result.m14 = 0.0f;
    result.m15 = 1.0f;

    return result;
}

// Returns xyz-rotation matrix (angles in radians)
RMDEF mat4 MatrixRotateXYZ(vec3 ang)
{
    mat4 result = MatrixIdentity();

    float cosz = cosf(-ang.z);
    float sinz = sinf(-ang.z);
    float cosy = cosf(-ang.y);
    float siny = sinf(-ang.y);
    float cosx = cosf(-ang.x);
    float sinx = sinf(-ang.x);

    result.m0 = cosz * cosy;
    result.m4 = (cosz * siny * sinx) - (sinz * cosx);
    result.m8 = (cosz * siny * cosx) + (sinz * sinx);

    result.m1 = sinz * cosy;
    result.m5 = (sinz * siny * sinx) + (cosz * cosx);
    result.m9 = (sinz * siny * cosx) - (cosz * sinx);

    result.m2 = -siny;
    result.m6 = cosy * sinx;
    result.m10= cosy * cosx;

    return result;
}

// Returns x-rotation matrix (angle in radians)
RMDEF mat4 MatrixRotateX(float angle)
{
    mat4 result = MatrixIdentity();

    float cosres = cosf(angle);
    float sinres = sinf(angle);

    result.m5 = cosres;
    result.m6 = -sinres;
    result.m9 = sinres;
    result.m10 = cosres;

    return result;
}

// Returns y-rotation matrix (angle in radians)
RMDEF mat4 MatrixRotateY(float angle)
{
    mat4 result = MatrixIdentity();

    float cosres = cosf(angle);
    float sinres = sinf(angle);

    result.m0 = cosres;
    result.m2 = sinres;
    result.m8 = -sinres;
    result.m10 = cosres;

    return result;
}

// Returns z-rotation matrix (angle in radians)
RMDEF mat4 MatrixRotateZ(float angle)
{
    mat4 result = MatrixIdentity();

    float cosres = cosf(angle);
    float sinres = sinf(angle);

    result.m0 = cosres;
    result.m1 = -sinres;
    result.m4 = sinres;
    result.m5 = cosres;

    return result;
}

// Returns scaling matrix
RMDEF mat4 MatrixScale(float x, float y, float z)
{
/*    mat4 result = { x, 0.0f, 0.0f, 0.0f,
                      0.0f, y, 0.0f, 0.0f,
                      0.0f, 0.0f, z, 0.0f,
                      0.0f, 0.0f, 0.0f, 1.0f };*/

    mat4 result = MatrixIdentity();

    result.m0 = x;
    result.m5 = y;
    result.m10 = z;

    return result;
}

#include <intrin.h>

// Returns two matrix multiplication
// NOTE: When multiplying matrices... the order matters!

/*
float m0, m4, m8, m12;
float m1, m5, m9, m13;
float m2, m6, m10, m14;
float m3, m7, m11, m15;
*/

// why is this slower??? memory?
RMDEF mat4 MatrixMultiplySIMD(mat4 right, mat4 left)
{
    vec4* lvec = (vec4*)(&left);
    mat4 out;
    vec4* ovec = (vec4*)(&out);

    __m128 e0 = _mm_load_ps((float*)(&right.m0));
    __m128 e1 = _mm_load_ps((float*)(&right.m4));
    __m128 e2 = _mm_load_ps((float*)(&right.m8));
    __m128 e3 = _mm_load_ps((float*)(&right.m12));

    for (int i = 0; i < 4; i++)
    {
        __m128 lv = _mm_load_ps((float*)&lvec[i]);

        __m128 lv0 = _mm_shuffle_ps(lv, lv, _MM_SHUFFLE(0, 0, 0, 0));
        __m128 lv1 = _mm_shuffle_ps(lv, lv, _MM_SHUFFLE(1, 1, 1, 1));
        __m128 lv2 = _mm_shuffle_ps(lv, lv, _MM_SHUFFLE(2, 2, 2, 2));
        __m128 lv3 = _mm_shuffle_ps(lv, lv, _MM_SHUFFLE(3, 3, 3, 3));

        __m128 m0 = _mm_mul_ps(lv0, e0);
        __m128 m1 = _mm_mul_ps(lv1, e1);
        __m128 m2 = _mm_mul_ps(lv2, e2);
        __m128 m3 = _mm_mul_ps(lv3, e3);

        __m128 a0 = _mm_add_ps(m0, m1);
        __m128 a1 = _mm_add_ps(m2, m3);
        __m128 a2 = _mm_add_ps(a0, a1);

        _mm_store_ps((float*)&ovec[i], a2);
    }

/*
    {
        __m128 lv = _mm_load_ps((float*)&lvec[1]);

        __m128 lv0 = _mm_shuffle_ps(lv, lv, _MM_SHUFFLE(0, 0, 0, 0));
        __m128 lv1 = _mm_shuffle_ps(lv, lv, _MM_SHUFFLE(1, 1, 1, 1));
        __m128 lv2 = _mm_shuffle_ps(lv, lv, _MM_SHUFFLE(2, 2, 2, 2));
        __m128 lv3 = _mm_shuffle_ps(lv, lv, _MM_SHUFFLE(3, 3, 3, 3));

        __m128 m0 = _mm_mul_ps(lv0, e0);
        __m128 m1 = _mm_mul_ps(lv1, e1);
        __m128 m2 = _mm_mul_ps(lv2, e2);
        __m128 m3 = _mm_mul_ps(lv3, e3);

        __m128 a0 = _mm_add_ps(m0, m1);
        __m128 a1 = _mm_add_ps(m2, m3);
        __m128 a2 = _mm_add_ps(a0, a1);

        _mm_store_ps((float*)&ovec[1], a2);
    }

    {
        __m128 lv = _mm_load_ps((float*)&lvec[2]);

        __m128 lv0 = _mm_shuffle_ps(lv, lv, _MM_SHUFFLE(0, 0, 0, 0));
        __m128 lv1 = _mm_shuffle_ps(lv, lv, _MM_SHUFFLE(1, 1, 1, 1));
        __m128 lv2 = _mm_shuffle_ps(lv, lv, _MM_SHUFFLE(2, 2, 2, 2));
        __m128 lv3 = _mm_shuffle_ps(lv, lv, _MM_SHUFFLE(3, 3, 3, 3));

        __m128 m0 = _mm_mul_ps(lv0, e0);
        __m128 m1 = _mm_mul_ps(lv1, e1);
        __m128 m2 = _mm_mul_ps(lv2, e2);
        __m128 m3 = _mm_mul_ps(lv3, e3);

        __m128 a0 = _mm_add_ps(m0, m1);
        __m128 a1 = _mm_add_ps(m2, m3);
        __m128 a2 = _mm_add_ps(a0, a1);

        _mm_store_ps((float*)&ovec[2], a2);
    }

    {
        __m128 lv = _mm_load_ps((float*)&lvec[3]);

        __m128 lv0 = _mm_shuffle_ps(lv, lv, _MM_SHUFFLE(0, 0, 0, 0));
        __m128 lv1 = _mm_shuffle_ps(lv, lv, _MM_SHUFFLE(1, 1, 1, 1));
        __m128 lv2 = _mm_shuffle_ps(lv, lv, _MM_SHUFFLE(2, 2, 2, 2));
        __m128 lv3 = _mm_shuffle_ps(lv, lv, _MM_SHUFFLE(3, 3, 3, 3));

        __m128 m0 = _mm_mul_ps(lv0, e0);
        __m128 m1 = _mm_mul_ps(lv1, e1);
        __m128 m2 = _mm_mul_ps(lv2, e2);
        __m128 m3 = _mm_mul_ps(lv3, e3);

        __m128 a0 = _mm_add_ps(m0, m1);
        __m128 a1 = _mm_add_ps(m2, m3);
        __m128 a2 = _mm_add_ps(a0, a1);

        _mm_store_ps((float*)&ovec[3], a2);
    }
*/

    return out;
}

RMDEF mat4 MatrixMultiply(mat4 right, mat4 left)
{
    mat4 result;

    result.m0  = left.m0*right.m0  + left.m1*right.m4  + left.m2*right.m8   + left.m3*right.m12;
    result.m1  = left.m0*right.m1  + left.m1*right.m5  + left.m2*right.m9   + left.m3*right.m13;
    result.m2  = left.m0*right.m2  + left.m1*right.m6  + left.m2*right.m10  + left.m3*right.m14;
    result.m3  = left.m0*right.m3  + left.m1*right.m7  + left.m2*right.m11  + left.m3*right.m15;

    result.m4  = left.m4*right.m0  + left.m5*right.m4  + left.m6*right.m8   + left.m7*right.m12;
    result.m5  = left.m4*right.m1  + left.m5*right.m5  + left.m6*right.m9   + left.m7*right.m13;
    result.m6  = left.m4*right.m2  + left.m5*right.m6  + left.m6*right.m10  + left.m7*right.m14;
    result.m7  = left.m4*right.m3  + left.m5*right.m7  + left.m6*right.m11  + left.m7*right.m15;

    result.m8  = left.m8*right.m0  + left.m9*right.m4  + left.m10*right.m8  + left.m11*right.m12;
    result.m9  = left.m8*right.m1  + left.m9*right.m5  + left.m10*right.m9  + left.m11*right.m13;
    result.m10 = left.m8*right.m2  + left.m9*right.m6  + left.m10*right.m10 + left.m11*right.m14;
    result.m11 = left.m8*right.m3  + left.m9*right.m7  + left.m10*right.m11 + left.m11*right.m15;

    result.m12 = left.m12*right.m0 + left.m13*right.m4 + left.m14*right.m8  + left.m15*right.m12;
    result.m13 = left.m12*right.m1 + left.m13*right.m5 + left.m14*right.m9  + left.m15*right.m13;
    result.m14 = left.m12*right.m2 + left.m13*right.m6 + left.m14*right.m10 + left.m15*right.m14;
    result.m15 = left.m12*right.m3 + left.m13*right.m7 + left.m14*right.m11 + left.m15*right.m15;

    return result;
}

RMDEF mat4 m4_mul_f(mat4 matrix, float scalar)
{
    mat4 result = MatrixIdentity();

    result.m0 = matrix.m0 * scalar;
    result.m1 = matrix.m1 * scalar;
    result.m2 = matrix.m2 * scalar;
    result.m3 = matrix.m3 * scalar;
    result.m4 = matrix.m4 * scalar;
    result.m5 = matrix.m5 * scalar;
    result.m6 = matrix.m6 * scalar;
    result.m7 = matrix.m7 * scalar;
    result.m8 = matrix.m8 * scalar;
    result.m9 = matrix.m9 * scalar;
    result.m10 = matrix.m10 * scalar;
    result.m11 = matrix.m11 * scalar;
    result.m12 = matrix.m12 * scalar;
    result.m13 = matrix.m13 * scalar;
    result.m14 = matrix.m14 * scalar;
    result.m15 = matrix.m15 * scalar;

    return result;
}

// Returns perspective projection matrix
RMDEF mat4 MatrixFrustum(double left, double right, double bottom, double top, double near, double far)
{
    mat4 result = { 0 };

    float rl = (float)(right - left);
    float tb = (float)(top - bottom);
    float fn = (float)(far - near);

    result.m0 = ((float) near*2.0f)/rl;
    result.m1 = 0.0f;
    result.m2 = 0.0f;
    result.m3 = 0.0f;

    result.m4 = 0.0f;
    result.m5 = ((float) near*2.0f)/tb;
    result.m6 = 0.0f;
    result.m7 = 0.0f;

    result.m8 = ((float)right + (float)left)/rl;
    result.m9 = ((float)top + (float)bottom)/tb;
    result.m10 = -((float)far + (float)near)/fn;
    result.m11 = -1.0f;

    result.m12 = 0.0f;
    result.m13 = 0.0f;
    result.m14 = -((float)far*(float)near*2.0f)/fn;
    result.m15 = 0.0f;

    return result;
}

// Returns perspective projection matrix
// NOTE: Angle should be provided in radians
RMDEF mat4 MatrixPerspective(double fovy, double aspect, double near, double far)
{
    double top = near*tan(fovy*0.5);
    double right = top*aspect;
    mat4 result = MatrixFrustum(-right, right, -top, top, near, far);

    return result;
}

// Returns orthographic projection matrix
RMDEF mat4 MatrixOrtho(double left, double right, double bottom, double top, double near, double far)
{
    mat4 result = { 0 };

    float rl = (float)(right - left);
    float tb = (float)(top - bottom);
    float fn = (float)(far - near);

    result.m0 = 2.0f/rl;
    result.m1 = 0.0f;
    result.m2 = 0.0f;
    result.m3 = 0.0f;
    result.m4 = 0.0f;
    result.m5 = 2.0f/tb;
    result.m6 = 0.0f;
    result.m7 = 0.0f;
    result.m8 = 0.0f;
    result.m9 = 0.0f;
    result.m10 = -2.0f/fn;
    result.m11 = 0.0f;
    result.m12 = -((float)left + (float)right)/rl;
    result.m13 = -((float)top + (float)bottom)/tb;
    result.m14 = -((float)far + (float)near)/fn;
    result.m15 = 1.0f;

    return result;
}

// Returns camera look-at matrix (view matrix)
RMDEF mat4 MatrixLookAt(vec3 eye, vec3 target, vec3 up)
{
    mat4 result = { 0 };

    vec3 z = Vector3Subtract(eye, target);
    z = Vector3Normalize(z);
    vec3 x = Vector3CrossProduct(up, z);
    x = Vector3Normalize(x);
    vec3 y = Vector3CrossProduct(z, x);
    y = Vector3Normalize(y);

    result.m0 = x.x;
    result.m1 = x.y;
    result.m2 = x.z;
    result.m3 = 0.0f;
    result.m4 = y.x;
    result.m5 = y.y;
    result.m6 = y.z;
    result.m7 = 0.0f;
    result.m8 = z.x;
    result.m9 = z.y;
    result.m10 = z.z;
    result.m11 = 0.0f;
    result.m12 = eye.x;
    result.m13 = eye.y;
    result.m14 = eye.z;
    result.m15 = 1.0f;

    result = MatrixInvert(result);

    return result;
}

RMDEF vec4 Vec4(float x, float y, float z, float w)
{
    //return (vec3){x,y,z};

    vec4 vec;
    vec.x = x;
    vec.y = y;
    vec.z = z;
    vec.w = w;
    return vec;
}

RMDEF vec4 Vector4MultiplyM(mat4 m, vec4 v)
{
    vec4 result;

    result.x = m.m0 * v.x + m.m4 * v.y + m.m8 * v.z + m.m12 * v.w;
    result.y = m.m1 * v.x + m.m5 * v.y + m.m9 * v.z + m.m13 * v.w;
    result.z = m.m2 * v.x + m.m6 * v.y + m.m10 * v.z + m.m14 * v.w;
    result.w = m.m3 * v.x + m.m7 * v.y + m.m11 * v.z + m.m15 * v.w;

    return result;
}

RMDEF vec4 Vector4Divide(vec4 v, float s)
{
    //return (vec4){v.x / s, v.y / s, v.z / s, v.w / s};

    vec4 vec;
    vec.x = v.x / s;
    vec.y = v.y / s;
    vec.z = v.z / s;
    vec.w = v.w / s;
    return vec;
}

RMDEF vec3 MatrixProject(vec3 obj, mat4 view, mat4 proj, vec4 viewport)
{
    vec4 tmp = {obj.x, obj.y, obj.z, 1};

    tmp = Vector4MultiplyM(view, tmp);
    tmp = Vector4MultiplyM(proj, tmp);

    tmp = Vector4Divide(tmp, tmp.w);

    tmp.x = tmp.x * 0.5f + 0.5f;
    tmp.y = tmp.y * 0.5f + 0.5f;

    tmp.x = tmp.x * viewport.z + viewport.x;
    tmp.y = tmp.y * viewport.w + viewport.y;

    //return (vec3){tmp.x, tmp.y, tmp.z};

    vec3 vec;
    vec.x = tmp.x;
    vec.y = tmp.y;
    vec.z = tmp.z;
    return vec;
}

// Returns float array of matrix data
RMDEF float16 MatrixToFloatV(mat4 mat)
{
    float16 buffer = { 0 };

    buffer.v[0] = mat.m0;
    buffer.v[1] = mat.m1;
    buffer.v[2] = mat.m2;
    buffer.v[3] = mat.m3;
    buffer.v[4] = mat.m4;
    buffer.v[5] = mat.m5;
    buffer.v[6] = mat.m6;
    buffer.v[7] = mat.m7;
    buffer.v[8] = mat.m8;
    buffer.v[9] = mat.m9;
    buffer.v[10] = mat.m10;
    buffer.v[11] = mat.m11;
    buffer.v[12] = mat.m12;
    buffer.v[13] = mat.m13;
    buffer.v[14] = mat.m14;
    buffer.v[15] = mat.m15;

    return buffer;
}

//----------------------------------------------------------------------------------
// Module Functions Definition - quat math
//----------------------------------------------------------------------------------

// Returns identity quaternion
RMDEF quat QuaternionIdentity(void)
{
    quat result = { 0.0f, 0.0f, 0.0f, 1.0f };
    return result;
}

RMDEF quat Quat(float x, float y, float z, float w)
{
    quat result;

    result.x = x;
    result.y = y;
    result.z = z;
    result.w = w;

    return result;
}

// Computes the length of a quaternion
RMDEF float QuaternionLength(quat q)
{
    float result = (float)sqrtf(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w);
    return result;
}

// Normalize provided quaternion
RMDEF quat QuaternionNormalize(quat q)
{
    quat result = { 0 };

    float length, ilength;
    length = QuaternionLength(q);
    if (length == 0.0f) length = 1.0f;
    ilength = 1.0f/length;

    result.x = q.x*ilength;
    result.y = q.y*ilength;
    result.z = q.z*ilength;
    result.w = q.w*ilength;

    return result;
}

// Invert provided quaternion
RMDEF quat QuaternionInvert(quat q)
{
    quat result = q;
    float length = QuaternionLength(q);
    float lengthSq = length*length;

    if (lengthSq != 0.0)
    {
        float i = 1.0f/lengthSq;

        result.x *= -i;
        result.y *= -i;
        result.z *= -i;
        result.w *= i;
    }

    return result;
}

// Calculate two quaternion multiplication
RMDEF quat QuaternionMultiply(quat q1, quat q2)
{
    quat result = { 0 };

    float qax = q1.x, qay = q1.y, qaz = q1.z, qaw = q1.w;
    float qbx = q2.x, qby = q2.y, qbz = q2.z, qbw = q2.w;

    result.x = qax*qbw + qaw*qbx + qay*qbz - qaz*qby;
    result.y = qay*qbw + qaw*qby + qaz*qbx - qax*qbz;
    result.z = qaz*qbw + qaw*qbz + qax*qby - qay*qbx;
    result.w = qaw*qbw - qax*qbx - qay*qby - qaz*qbz;

    return result;
}

// Calculate linear interpolation between two quaternions
RMDEF quat QuaternionLerp(quat q1, quat q2, float amount)
{
    quat result = { 0 };

    result.x = q1.x + amount*(q2.x - q1.x);
    result.y = q1.y + amount*(q2.y - q1.y);
    result.z = q1.z + amount*(q2.z - q1.z);
    result.w = q1.w + amount*(q2.w - q1.w);

    return result;
}

// Calculate slerp-optimized interpolation between two quaternions
RMDEF quat QuaternionNlerp(quat q1, quat q2, float amount)
{
    quat result = QuaternionLerp(q1, q2, amount);
    result = QuaternionNormalize(result);

    return result;
}

// Calculates spherical linear interpolation between two quaternions
RMDEF quat QuaternionSlerp(quat q1, quat q2, float amount)
{
    quat result = { 0 };

    float cosHalfTheta =  q1.x*q2.x + q1.y*q2.y + q1.z*q2.z + q1.w*q2.w;

    if (cosHalfTheta < 0)
    {
        q1.x = -q1.x;
        q1.y = -q1.y;
        q1.z = -q1.z;
        q1.w = -q1.w;

        cosHalfTheta = -cosHalfTheta;
    }

    if (fabs(cosHalfTheta) >= 1.0f)
        result = q1;
    else if (cosHalfTheta > 0.95f)
        result = QuaternionNlerp(q1, q2, amount);
    else
    {
        float halfTheta = (float) acos(cosHalfTheta);
        float sinHalfTheta = (float) sqrtf(1.0f - cosHalfTheta*cosHalfTheta);

        if (fabs(sinHalfTheta) < 0.001f)
        {
            result.x = (q1.x*0.5f + q2.x*0.5f);
            result.y = (q1.y*0.5f + q2.y*0.5f);
            result.z = (q1.z*0.5f + q2.z*0.5f);
            result.w = (q1.w*0.5f + q2.w*0.5f);
        }
        else
        {
            float ratioA = sinf((1 - amount)*halfTheta)/sinHalfTheta;
            float ratioB = sinf(amount*halfTheta)/sinHalfTheta;

            result.x = (q1.x*ratioA + q2.x*ratioB);
            result.y = (q1.y*ratioA + q2.y*ratioB);
            result.z = (q1.z*ratioA + q2.z*ratioB);
            result.w = (q1.w*ratioA + q2.w*ratioB);
        }
    }

    return result;
}

// Calculate quaternion based on the rotation from one vector to another
RMDEF quat QuaternionFromVector3ToVector3(vec3 from, vec3 to)
{
    quat result = { 0 };

    float cos2Theta = Vector3DotProduct(from, to);
    vec3 cross = Vector3CrossProduct(from, to);

    result.x = cross.x;
    result.y = cross.y;
    result.z = cross.y;
    result.w = 1.0f + cos2Theta;     // NOTE: Added QuaternioIdentity()

    // Normalize to essentially nlerp the original and identity to 0.5
    result = QuaternionNormalize(result);

    // Above lines are equivalent to:
    //quat result = QuaternionNlerp(q, QuaternionIdentity(), 0.5f);

    return result;
}

// Returns a quaternion for a given rotation matrix
RMDEF quat QuaternionFromMatrix(mat4 mat)
{
    quat result = { 0 };

    float trace = MatrixTrace(mat);

    if (trace > 0.0f)
    {
        float s = (float)sqrtf(trace + 1)*2.0f;
        float invS = 1.0f/s;

        result.w = s*0.25f;
        result.x = (mat.m6 - mat.m9)*invS;
        result.y = (mat.m8 - mat.m2)*invS;
        result.z = (mat.m1 - mat.m4)*invS;
    }
    else
    {
        float m00 = mat.m0, m11 = mat.m5, m22 = mat.m10;

        if (m00 > m11 && m00 > m22)
        {
            float s = (float)sqrtf(1.0f + m00 - m11 - m22)*2.0f;
            float invS = 1.0f/s;

            result.w = (mat.m6 - mat.m9)*invS;
            result.x = s*0.25f;
            result.y = (mat.m4 + mat.m1)*invS;
            result.z = (mat.m8 + mat.m2)*invS;
        }
        else if (m11 > m22)
        {
            float s = (float)sqrtf(1.0f + m11 - m00 - m22)*2.0f;
            float invS = 1.0f/s;

            result.w = (mat.m8 - mat.m2)*invS;
            result.x = (mat.m4 + mat.m1)*invS;
            result.y = s*0.25f;
            result.z = (mat.m9 + mat.m6)*invS;
        }
        else
        {
            float s = (float)sqrtf(1.0f + m22 - m00 - m11)*2.0f;
            float invS = 1.0f/s;

            result.w = (mat.m1 - mat.m4)*invS;
            result.x = (mat.m8 + mat.m2)*invS;
            result.y = (mat.m9 + mat.m6)*invS;
            result.z = s*0.25f;
        }
    }

    return result;
}

// Returns a matrix for a given quaternion
RMDEF mat4 QuaternionToMatrix(quat q)
{
    mat4 result = { 0 };

    quat normed = QuaternionNormalize(q);

    float xx, yy, zz,
          xy, xz, yz,
          wx, wy, wz;

    xx = normed.x * normed.x;
    yy = normed.y * normed.y;
    zz = normed.z * normed.z;
    xy = normed.x * normed.y;
    xz = normed.x * normed.z;
    yz = normed.y * normed.z;
    wx = normed.w * normed.x;
    wy = normed.w * normed.y;
    wz = normed.w * normed.z;

    result.m0 = 1.0f - 2.0f * (yy + zz);
    result.m1 = 2.0f * (xy + wz);
    result.m2 = 2.0f * (xz - wy);
    result.m3 = 0.0f;

    result.m4 = 2.0f * (xy - wz);
    result.m5 = 1.0f - 2.0f * (xx + zz);
    result.m6 = 2.0f * (yz + wx);
    result.m7 = 0.0f;

    result.m8 = 2.0f * (xz + wy);
    result.m9 = 2.0f * (yz - wx);
    result.m10 = 1.0f - 2.0f * (xx + yy);
    result.m11 = 0.0f;

    result.m12 = 0.0f;
    result.m13 = 0.0f;
    result.m14 = 0.0f;
    result.m15 = 1.0f;

    return result;
}

// Returns rotation quaternion for an angle and axis
// NOTE: angle must be provided in radians
RMDEF quat QuaternionFromAxisAngle(vec3 axis, float angle)
{
    quat result = { 0.0f, 0.0f, 0.0f, 1.0f };

    if (Vector3Length(axis) != 0.0f)

    angle *= 0.5f;

    axis = Vector3Normalize(axis);

    float sinres = sinf(angle);
    float cosres = cosf(angle);

    result.x = axis.x*sinres;
    result.y = axis.y*sinres;
    result.z = axis.z*sinres;
    result.w = cosres;

    result = QuaternionNormalize(result);

    return result;
}

// Returns the rotation angle and axis for a given quaternion
RMDEF void QuaternionToAxisAngle(quat q, vec3 *outAxis, float *outAngle)
{
    if (fabs(q.w) > 1.0f) q = QuaternionNormalize(q);

    vec3 resAxis = { 0.0f, 0.0f, 0.0f };
    float resAngle = 0.0f;

    resAngle = 2.0f*(float)acos(q.w);
    float den = (float)sqrtf(1.0f - q.w*q.w);

    if (den > 0.0001f)
    {
        resAxis.x = q.x/den;
        resAxis.y = q.y/den;
        resAxis.z = q.z/den;
    }
    else
    {
        // This occurs when the angle is zero.
        // Not a problem: just set an arbitrary normalized axis.
        resAxis.x = 1.0f;
    }

    *outAxis = resAxis;
    *outAngle = resAngle;
}

// Returns he quaternion equivalent to Euler angles
RMDEF quat QuaternionFromEuler(float roll, float pitch, float yaw)
{
    quat q = { 0 };

    float x0 = cosf(roll*0.5f);
    float x1 = sinf(roll*0.5f);
    float y0 = cosf(pitch*0.5f);
    float y1 = sinf(pitch*0.5f);
    float z0 = cosf(yaw*0.5f);
    float z1 = sinf(yaw*0.5f);

    q.x = x1*y0*z0 - x0*y1*z1;
    q.y = x0*y1*z0 + x1*y0*z1;
    q.z = x0*y0*z1 - x1*y1*z0;
    q.w = x0*y0*z0 + x1*y1*z1;

    return q;
}

// Return the Euler angles equivalent to quaternion (roll, pitch, yaw)
// NOTE: Angles are returned in a vec3 struct in degrees
RMDEF vec3 QuaternionToEuler(quat q)
{
    vec3 result = { 0 };

    // roll (x-axis rotation)
    float x0 = 2.0f*(q.w*q.x + q.y*q.z);
    float x1 = 1.0f - 2.0f*(q.x*q.x + q.y*q.y);
    result.x = atan2f(x0, x1)*RAD2DEG;

    // pitch (y-axis rotation)
    float y0 = 2.0f*(q.w*q.y - q.z*q.x);
    y0 = y0 > 1.0f ? 1.0f : y0;
    y0 = y0 < -1.0f ? -1.0f : y0;
    result.y = asinf(y0)*RAD2DEG;

    // yaw (z-axis rotation)
    float z0 = 2.0f*(q.w*q.z + q.x*q.y);
    float z1 = 1.0f - 2.0f*(q.y*q.y + q.z*q.z);
    result.z = atan2f(z0, z1)*RAD2DEG;

    return result;
}

// Transform a quaternion given a transformation matrix
RMDEF quat QuaternionTransform(quat q, mat4 mat)
{
    quat result = { 0 };

    result.x = mat.m0*q.x + mat.m4*q.y + mat.m8*q.z + mat.m12*q.w;
    result.y = mat.m1*q.x + mat.m5*q.y + mat.m9*q.z + mat.m13*q.w;
    result.z = mat.m2*q.x + mat.m6*q.y + mat.m10*q.z + mat.m14*q.w;
    result.w = mat.m3*q.x + mat.m7*q.y + mat.m11*q.z + mat.m15*q.w;

    return result;
}

#endif  // RAYMATH_H
