#ifndef SPHSOLVERCUDA_H
#define SPHSOLVERCUDA_H

//----------------------------------------------------------------------------------------------------------------------
/// @file SPHSolver.h
/// @brief Calculates and updates our new particle positions with navier-stokes equations using CUDA acceleration.
/// @author Declan Russell
/// @version 1.0
/// @date 03/02/2015
/// @class SPHSolverCUDA
//----------------------------------------------------------------------------------------------------------------------

#ifdef DARWIN
    #include <OpenGL/gl3.h>
#else
    #include <GL/glew.h>
    #ifndef WIN32
        #include <GL/gl.h>
    #endif
#endif

// Just another stupid quirk by windows -.-
// Without windows.h defined before cuda_gl_interop you get
// redefinition conflicts.
#ifdef WIN32
    #include <Windows.h>
#endif
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cuda_gl_interop.h>

#include "SPHSolverCUDAKernals.h"
#include <vector>

class SPHSolverCUDA
{
public:
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Our defualt constructor
    /// @param _x - the x boundary of our simulation
    /// @param _y - the y boundary of our simulation
    /// @param _t - the thickness of our boundary
    /// @param _l - the number of layers we want in our boundary
    //----------------------------------------------------------------------------------------------------------------------
    SPHSolverCUDA(float _x = 15.f, float _y = 15.f, float _t = 0.05, float _l = 3);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief destructor
    //----------------------------------------------------------------------------------------------------------------------
    ~SPHSolverCUDA();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Sets the particles init positions in our simulation from an array. If set more than once old data will be removed.
    //----------------------------------------------------------------------------------------------------------------------
    void setParticles(std::vector<float3> &_particles);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Retrieves our particles from the GPU and returns them in a vector
    /// @return array of particle positions (vector<float3>)
    //----------------------------------------------------------------------------------------------------------------------
    std::vector<float3> getParticlePositions();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Generates a defined number of random positions for our simulation.
    /// @brief Note that these samples will replace the original samples in the simulation.
    /// @param _n - number of samples to generate.
    //----------------------------------------------------------------------------------------------------------------------
    void genRandomSamples(float _n);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Retrieves our particles from the GPU and returns them in a vector in float2 form
    /// @return array of particle positions (vector<float3>)
    //----------------------------------------------------------------------------------------------------------------------
    std::vector<float2> getParticlePosF2();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief accessor to the number of particles in our simulation
    //----------------------------------------------------------------------------------------------------------------------
    inline int getNumParticles(){return m_simProperties.numParticles;}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief accessor to the number of boundary particles in our simulation
    //----------------------------------------------------------------------------------------------------------------------
    inline int getNumBoundParticles(){return m_numBoundParticles;}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Returns our OpenGL VAO handle to our particle positions
    /// @return OpenGL VAO handle to our particle positions (GLuint)
    //----------------------------------------------------------------------------------------------------------------------
    inline GLuint getPositionsVAO(){return m_posVAO;}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief returns our OpenGL VAO handle to our boundary particle positions
    /// @return OpenGL VAO handle to our particle positions (GLuint)
    //----------------------------------------------------------------------------------------------------------------------
    inline GLuint getBndPositionsVAO(){return m_bndPosVAO;}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief set the mass of our particles
    /// @param _m - mass of our particles (float)
    //----------------------------------------------------------------------------------------------------------------------
    inline void setMass(float _m){m_simProperties.mass = _m; updateGPUSimProps();}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief accessor to the mass of our particles
    //----------------------------------------------------------------------------------------------------------------------
    inline float getMass(){return m_simProperties.mass;}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief mutator for the timestep of our simulation
    /// @param _t - desired timestep
    //----------------------------------------------------------------------------------------------------------------------
    inline void setTimeStep(float _t){m_simProperties.timeStep = _t; updateGPUSimProps();}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief mutator to k our gas/stiffness constant
    /// @param _k - desired gas/stiffness constant
    //----------------------------------------------------------------------------------------------------------------------
    inline void setKConst(float _k){m_simProperties.k = _k; updateGPUSimProps();}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief accessor to k our gas/stiffness constant
    /// @return k our gas/stiffness constant (float)
    //----------------------------------------------------------------------------------------------------------------------
    inline float getKConst(){return m_simProperties.k;}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief mutator to our smoothing length h
    /// @param _h - desired smoothing length
    //----------------------------------------------------------------------------------------------------------------------
    inline void setSmoothingLength(float _h);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief mutator to our rest/target density
    /// @param _d - desired rest/target density
    //----------------------------------------------------------------------------------------------------------------------
    inline void setRestDensity(float _d){m_restDensity = _d; updateGPUSimProps();}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief updates the sim properties on our GPU
    //----------------------------------------------------------------------------------------------------------------------
    inline void updateGPUSimProps(){updateSimProps(&m_simProperties);}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief our update function to increment the step of our simulation
    //----------------------------------------------------------------------------------------------------------------------
    void update();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief compute the average density of our simulation
    /// @return average density of simulation (float)
    //----------------------------------------------------------------------------------------------------------------------
    inline float getAverageDensity(){return computeAverageDensity(m_simProperties.numParticles,m_fluidBuffers);}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief mutator to our density difference
    /// @param _diff - desired density difference
    //----------------------------------------------------------------------------------------------------------------------
    inline void setDensityDiff(float _diff){m_densityDiff = _diff;}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief mutator to our convergence value
    /// @param _x - desired convergence value (int)
    //----------------------------------------------------------------------------------------------------------------------
    inline void setConvergeValue(int _x){m_simProperties.convergeValue = _x; updateGPUSimProps();}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief accessor to our convergence value
    /// @return convergence value (float)
    //----------------------------------------------------------------------------------------------------------------------
    inline float getConvergeValue(){return m_simProperties.convergeValue;}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief checks to see if our simulation has convered
    /// @return is our simulation has convereged (bool)
    //----------------------------------------------------------------------------------------------------------------------
    inline bool convergedState(){return isConverged(m_simProperties.numParticles,m_fluidBuffers);}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief sets the pixel varience of our image
    /// @param _ptr - pixel intensity information (std::vector<float>)
    //----------------------------------------------------------------------------------------------------------------------
    void setPixelVariance(std::vector<float> &_i);
    //----------------------------------------------------------------------------------------------------------------------
private:
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief the boundaries of our simulation
    //----------------------------------------------------------------------------------------------------------------------
    float3 m_simBounds;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief volume of our fluid
    //----------------------------------------------------------------------------------------------------------------------
    float m_volume;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief the number of boundary particles we have
    //----------------------------------------------------------------------------------------------------------------------
    int m_numBoundParticles;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Our maximum threads per block.
    //----------------------------------------------------------------------------------------------------------------------
    int m_threadsPerBlock;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief VAO handle of our positions buffer
    //----------------------------------------------------------------------------------------------------------------------
    GLuint m_posVAO;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief VBO handle to our positions buffer
    //----------------------------------------------------------------------------------------------------------------------
    GLuint m_posVBO;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief VAO handle for our boundary positions buffer
    //----------------------------------------------------------------------------------------------------------------------
    GLuint m_bndPosVAO;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief VBO handle to our boundary positions buffer
    //----------------------------------------------------------------------------------------------------------------------
    GLuint m_bndVBO;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief our cuda graphics resource for our particle positions OpenGL interop.
    //----------------------------------------------------------------------------------------------------------------------
    cudaGraphicsResource_t m_resourcePos;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief our cuda graphics resource for our boundary particles positions OpenGL interop.
    //----------------------------------------------------------------------------------------------------------------------
    cudaGraphicsResource_t m_resourceBndPos;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief our fluid buffers on our device
    //----------------------------------------------------------------------------------------------------------------------
    fluidBuffers m_fluidBuffers;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Structure to hold all of our simulation properties so we can easily pass it to our CUDA kernal.
    //----------------------------------------------------------------------------------------------------------------------
    SimProps m_simProperties;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Our CUDA stream to help run kernals concurrently
    //----------------------------------------------------------------------------------------------------------------------
    cudaStream_t m_cudaStream;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Rest/Target density of our fluid
    //----------------------------------------------------------------------------------------------------------------------
    float m_restDensity;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief the density difference of our simulation
    //----------------------------------------------------------------------------------------------------------------------
    float m_densityDiff;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief function to set our hash grid position and dimensions
    /// @param _gridMin - minimum position of our grid
    /// @param _gridDim - grid dimentions
    //----------------------------------------------------------------------------------------------------------------------
    void setHashPosAndDim(float2 _gridMin, float2 _gridDim);
    //----------------------------------------------------------------------------------------------------------------------

};

#endif // SPHSOLVERKERNALS_H

