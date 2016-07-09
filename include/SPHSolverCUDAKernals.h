#ifndef SPHSOLVERCUDAKERNALS
#define SPHSOLVERCUDAKERNALS

#include <helper_math.h>
#include <cuda_runtime.h>

//----------------------------------------------------------------------------------------------------------------------
/// @breif Structure to hold all our simulation properties for easy passing to our kernals
//----------------------------------------------------------------------------------------------------------------------
struct SimProps
{
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief constant used in our density weighting kernal. Faster if we precalcuate is and store it.
    //----------------------------------------------------------------------------------------------------------------------
    float dWConst;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief constant used in our pressure weighting kernal. Faster if we precalcuate is and store it.
    //----------------------------------------------------------------------------------------------------------------------
    float pWConst;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief constant used in our viscosity weighting kernal. Faster if we precalcuate is and store it.
    //----------------------------------------------------------------------------------------------------------------------
    float vWConst;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief 1st constant used in our cohesion weighting kernal. Faster if we precalculate it and store it.
    //----------------------------------------------------------------------------------------------------------------------
    float cWConst1;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief 2nd constant used in our cohesion weighting kernal. Faster if we precalculate it and store it.
    //----------------------------------------------------------------------------------------------------------------------
    float cWConst2;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Smoothing length of our simulation
    //----------------------------------------------------------------------------------------------------------------------
    float h;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Smoothing length squared
    //----------------------------------------------------------------------------------------------------------------------
    float hSqrd;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Gas constant of our simualtion. Can also be thought of as stiffness. This default to (Speed of sound)^2.
    /// @brief Smaller value lets fluid compress more and vice versa.
    //----------------------------------------------------------------------------------------------------------------------
    float k;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief the timestep of our simulation
    //----------------------------------------------------------------------------------------------------------------------
    float timeStep;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief the gravity force in our simulation
    //----------------------------------------------------------------------------------------------------------------------
    float3 gravity;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief the number of particles in our simulation
    //----------------------------------------------------------------------------------------------------------------------
    int numParticles;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief particle mass
    //----------------------------------------------------------------------------------------------------------------------
    float mass;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Surface tension coeficient of our simulation
    //----------------------------------------------------------------------------------------------------------------------
    float tension;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief our hash grid minimum
    //----------------------------------------------------------------------------------------------------------------------
    float2 gridMin;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief our hash grid dimentions
    //----------------------------------------------------------------------------------------------------------------------
    float2 gridDim;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief our grid resolution
    //----------------------------------------------------------------------------------------------------------------------
    int2 gridRes;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief minimum value of change it reach our convergence condition
    //----------------------------------------------------------------------------------------------------------------------
    float convergeValue;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief acceleration limit
    //----------------------------------------------------------------------------------------------------------------------
    float accLimit;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief accerleration limit squared
    //----------------------------------------------------------------------------------------------------------------------
    float accLimit2;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief velocity limit
    //----------------------------------------------------------------------------------------------------------------------
    float velLimit;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief velocity limit squared
    //----------------------------------------------------------------------------------------------------------------------
    float velLimit2;
    //----------------------------------------------------------------------------------------------------------------------
};
//----------------------------------------------------------------------------------------------------------------------
/// @brief Structure to hold our neighbour cell locations
//----------------------------------------------------------------------------------------------------------------------
struct cellInfo
{
    // Number of neighbouring cells we have
    int cNum;
    // Idx's of neighbouring cells
    int cIdx[9];
};
//----------------------------------------------------------------------------------------------------------------------
/// @brief Structure to hold our fluid buffers
//----------------------------------------------------------------------------------------------------------------------
struct fluidBuffers
{
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief pointer to our position buffer on our device
    //----------------------------------------------------------------------------------------------------------------------
    float3 *posPtr;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief pointer to our velocity buffer on our device
    //----------------------------------------------------------------------------------------------------------------------
    float3 *velPtr;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief pointer to our acceleration buffer on our device
    //----------------------------------------------------------------------------------------------------------------------
    float3 *accPtr;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief pointer to our density buffer on our device
    //----------------------------------------------------------------------------------------------------------------------
    float *denPtr;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief pointer to our pixel variance buffer
    //----------------------------------------------------------------------------------------------------------------------
    float* pixelVar;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief pointer to our cell occupancy buffer on our device
    //----------------------------------------------------------------------------------------------------------------------
    int *cellOccBuffer;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief a buffer that holds the index's of where the points in our hash cell begin.
    //----------------------------------------------------------------------------------------------------------------------
    int *cellIndexBuffer;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief pointer to our hash keys buffer on our device
    //----------------------------------------------------------------------------------------------------------------------
    int *hashKeys;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief pointer to our hash map buffer on our device
    //----------------------------------------------------------------------------------------------------------------------
    cellInfo *hashMap;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief pointer to our buffer that holds if our particles have reached our converged condition
    //----------------------------------------------------------------------------------------------------------------------
    int *convergedPtr;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief a buffer that holds the cell occupancy of our boundary particles
    //----------------------------------------------------------------------------------------------------------------------
    int *bndCellOccBuff;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief a buffer that holds the index's of where the points in our boundary hash begin
    //----------------------------------------------------------------------------------------------------------------------
    int *bndCellIdxBuff;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief our boundary particle positions
    //----------------------------------------------------------------------------------------------------------------------
    float3 *bndPos;
    //----------------------------------------------------------------------------------------------------------------------
};
//----------------------------------------------------------------------------------------------------------------------
/// @brief just a test function to see if CUDA is working.
//----------------------------------------------------------------------------------------------------------------------
void test();
//----------------------------------------------------------------------------------------------------------------------
/// @brief Computes the average density of our particles
/// @param _numParticles - number of particles in our simulation
/// @param _buff - our simulation device buffers
/// @return average density of our particles (float)
//----------------------------------------------------------------------------------------------------------------------
float computeAverageDensity(int _numParticles, fluidBuffers _buff);
//----------------------------------------------------------------------------------------------------------------------
/// @brief updates our simulation properties on our GPU
/// @param _props - pointer to our simlulation properties
//----------------------------------------------------------------------------------------------------------------------
void updateSimProps(SimProps *_props);
//----------------------------------------------------------------------------------------------------------------------
/// @brief fills a buffer of ints with zeros
/// @brief _stream - Cuda stream to run our kernal on.
/// @param _threadsPerBlock - number of threads we have availible per block.
/// @param _bufferPtr - pointer to our target buffer
/// @param _size - size of our buffer
//----------------------------------------------------------------------------------------------------------------------
void fillIntZero(cudaStream_t _stream, int _threadsPerBlock, int *_bufferPtr, int size);
//----------------------------------------------------------------------------------------------------------------------
/// @brief Creates our hash table map so we know all cells neighbours
/// @param _stream - Cuda stream to run our kernal on.
/// @param _threadsPerBlock - number of threads we have availible per block.
/// @param _hashTableSize - size of our hash table
/// @param _buff - our simualtion device buffers
//----------------------------------------------------------------------------------------------------------------------
void createHashMap(cudaStream_t _stream, int _threadsPerBlock, int _hashTableSize, fluidBuffers _buff);
//----------------------------------------------------------------------------------------------------------------------
/// @brief Our hash and sort function for particles that only need to be hashed once that also have variance. E.g. Frozen paricles.
/// @param _stream - Cuda stream to run our kernal on.
/// @param _threadsPerBlock - number of threads we have availible per block.
/// @param _numParticles - numbder of particles in our sim
/// @param _hashTableSize - size of our hash table
/// @param _posPtr - pointer to our position buffer
/// @param _varPtr - pointer to our variance buffer
/// @param _occPtr - pointer to our occupancy buffer
/// @param _idxPtr - pointer to our index buffer
//----------------------------------------------------------------------------------------------------------------------
void hashAndSortFzn(cudaStream_t _stream, int _threadsPerBlock, int _numParticles, int _hashTableSize, float3 *posPtr, float *_varPtr, int *_occPtr, int *_idxPtr);
//----------------------------------------------------------------------------------------------------------------------
/// @brief Our hash and sort function for particles that only need to be hashed once. E.g. Boundary paricles.
/// @param _stream - Cuda stream to run our kernal on.
/// @param _threadsPerBlock - number of threads we have availible per block.
/// @param _numParticles - numbder of particles in our sim
/// @param _hashTableSize - size of our hash table
/// @param _posPtr - pointer to our position buffer
/// @param _occPtr - pointer to our occupancy buffer
/// @param _idxPtr - pointer to our index buffer
//----------------------------------------------------------------------------------------------------------------------
void hashAndSortBnd(cudaStream_t _stream, int _threadsPerBlock, int _numParticles, int _hashTableSize, float3 *posPtr, int *_occPtr, int *_idxPtr);
//----------------------------------------------------------------------------------------------------------------------
/// @brief Our hash and sort function
/// @param _stream - Cuda stream to run our kernal on.
/// @param _threadsPerBlock - number of threads we have availible per block.
/// @param _numParticles - numbder of particles in our sim
/// @param _hashTableSize - size of our hash table
/// @param _buff - our simualtion device buffers
//----------------------------------------------------------------------------------------------------------------------
void hashAndSort(cudaStream_t _stream, int _threadsPerBlock, int _numParticles, int _hashTableSize, fluidBuffers _buff);
//----------------------------------------------------------------------------------------------------------------------
/// @brief Our fluid solver function. Solves for our particles new positions through our navier stokes technique.
/// @param _stream - Cuda stream to run our kernal on.
/// @param _threadsPerBlock - number of threads we have availible per block.
/// @param _numParticles - number of particles in our sim
/// @param _buff - our simualtion device buffers
//----------------------------------------------------------------------------------------------------------------------
void initDensity(cudaStream_t _stream, int _threadsPerBlock, int _numParticles, fluidBuffers _buff);
//----------------------------------------------------------------------------------------------------------------------
/// @brief Our fluid solver function. Solves for our particles new positions through our navier stokes technique.
/// @param _stream - Cuda stream to run our kernal on.
/// @param _threadsPerBlock - number of threads we have availible per block.
/// @param _numParticles - numbder of particles in our sim
/// @param _restDensity - the rest density of our particles
/// @param _buff - our simualtion device buffers
//----------------------------------------------------------------------------------------------------------------------
void solve(cudaStream_t _stream, int _threadsPerBlock, int _numParticles, float _restDensity, fluidBuffers _buff);
//----------------------------------------------------------------------------------------------------------------------
/// @brief Function to check to see if our simulation has converged
/// @param _stream - Cuda stream to run our kernal on.
/// @param _threadsPerBlock - number of threads we have availible per block.
/// @param _restDensity - the rest density of our particles
/// @param _buff - our simualtion device buffers
//----------------------------------------------------------------------------------------------------------------------
bool isConverged(int _numParticles, fluidBuffers _buff);
//----------------------------------------------------------------------------------------------------------------------


#endif // SPHSOLVERCUDAKERNALS

