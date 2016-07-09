#include "SPHSolverCUDA.h"
#include <iostream>
#define SpeedOfSound 34.29f
#include <helper_math.h>
#include <ctime>
#include <boost/random.hpp>
#include <boost/generator_iterator.hpp>

//----------------------------------------------------------------------------------------------------------------------
SPHSolverCUDA::SPHSolverCUDA(float _x, float _y, float _t, float _l)
{
    //Lets test some cuda stuff
    int count;
    if (cudaGetDeviceCount(&count))
        return;
    std::cout << "Found " << count << " CUDA device(s)" << std::endl;
    if(count == 0){
        std::cerr<<"Install an Nvidia chip!"<<std::endl;
        exit(-1);
    }
    for (int i=0; i < count; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout<<prop.name<<", Compute capability:"<<prop.major<<"."<<prop.minor<<std::endl;;
        std::cout<<"  Global mem: "<<prop.totalGlobalMem/ 1024 / 1024<<"M, Shared mem per block: "<<prop.sharedMemPerBlock / 1024<<"k, Registers per block: "<<prop.regsPerBlock<<std::endl;
        std::cout<<"  Warp size: "<<prop.warpSize<<" threads, Max threads per block: "<<prop.maxThreadsPerBlock<<", Multiprocessor count: "<<prop.multiProcessorCount<<" MaxBlocks: "<<prop.maxGridSize[0]<<std::endl;
        m_threadsPerBlock = prop.maxThreadsPerBlock;
    }

    // Create our CUDA stream to run our kernals on. This helps with running kernals concurrently.
    // Check them out at http://on-demand.gputechconf.com/gtc-express/2011/presentations/StreamsAndConcurrencyWebinar.pdf
    checkCudaErrors(cudaStreamCreate(&m_cudaStream));

    // Make sure these are init to 0
    m_fluidBuffers.accPtr = 0;
    m_fluidBuffers.velPtr = 0;
    m_fluidBuffers.cellIndexBuffer = 0;
    m_fluidBuffers.cellOccBuffer = 0;
    m_fluidBuffers.hashKeys = 0;
    m_fluidBuffers.hashMap = 0;
    m_fluidBuffers.denPtr = 0;
    m_fluidBuffers.convergedPtr = 0;
    m_fluidBuffers.bndCellIdxBuff = 0;
    m_fluidBuffers.bndCellOccBuff = 0;
    m_fluidBuffers.pixelI = 0;
    m_fluidBuffers.pixelCMYK = 0;
    m_fluidBuffers.classBuff = 0;

    m_simBounds = make_float3(_x,_y,0.f);
    std::vector<float> intensity;
    intensity.resize(200*200);
    std::vector<float4> cmyk;
    cmyk.resize(200*200);
    for(unsigned int i=0;i<intensity.size();i++)
    {
        intensity[i]=1.f;
        cmyk[i] = make_float4(1.f,1.f,1.f,1.f);
    }
    checkCudaErrors(cudaMalloc(&m_fluidBuffers.pixelI,sizeof(float)*intensity.size()));
    checkCudaErrors(cudaMemcpy(m_fluidBuffers.pixelI,&intensity[0],sizeof(float)*intensity.size(),cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc(&m_fluidBuffers.pixelCMYK,sizeof(float4)*cmyk.size()));
    checkCudaErrors(cudaMemcpy(m_fluidBuffers.pixelCMYK,&cmyk[0],sizeof(float)*cmyk.size(),cudaMemcpyHostToDevice));

    m_simProperties.gridDim = make_float2(0,0);
    setSmoothingLength(0.3f);
    m_simProperties.timeStep = 0.001f;
    m_simProperties.gravity = make_float3(0.f,-9.8f,0.f);
    m_simProperties.convergeValue = 0.001f;
    m_simProperties.k = SpeedOfSound*20.f;//SpeedOfSound;
    m_simProperties.mass = 0.1f;
    m_simProperties.tension = 0.f;//1.f;
    m_simProperties.accLimit = 15.f;
    m_simProperties.accLimit2 = m_simProperties.accLimit*m_simProperties.accLimit;
    m_simProperties.velLimit = 1.f;
    m_simProperties.velLimit2 = m_simProperties.velLimit*m_simProperties.velLimit;
    m_restDensity = 500.f;
    m_densityDiff = 150.f;
    m_volume = 0;
    m_multiclass = false;

    //Define our boundaries
    float2 hmin = make_float2(-_t,-_t);
    float2 hmax = make_float2(_x+m_simProperties.h+_t,_y+m_simProperties.h+_t);
    float step = _t/_l;
    std::vector<float3> bndTemp;
    for (float x=-_t;x<_x+_t;x+=step)
    for (float y=-_t;y<_y+_t;y+=step)
        if (!((x>=0.f)&&(x<=_x) && (y>=0.f) && (y<=_y)))
            bndTemp.push_back(make_float3(x,y,0.f));

    m_numBoundParticles = (int)bndTemp.size();
    // Create an OpenGL buffer for our boundary position buffer
    // Create our VAO and vertex buffers
    glGenVertexArrays(1, &m_bndPosVAO);
    glBindVertexArray(m_bndPosVAO);

    // Put our vertices into an OpenGL buffer
    glGenBuffers(1, &m_bndVBO);
    glBindBuffer(GL_ARRAY_BUFFER, m_bndVBO);
    // We must alocate some space otherwise cuda cannot register it
    glBufferData(GL_ARRAY_BUFFER, sizeof(float3)*bndTemp.size(), &bndTemp[0], GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    // create our cuda graphics resource for our vertexs used for our OpenGL interop
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_resourceBndPos, m_bndVBO, cudaGraphicsRegisterFlagsWriteDiscard));

    // Unbind everything just in case
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Create our hash table
    setHashPosAndDim(hmin,hmax);

    // Hash and sort our boundary particles
    // Map our pointer for our position data
    size_t posSize;
    checkCudaErrors(cudaGraphicsMapResources(1,&m_resourceBndPos));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&m_fluidBuffers.bndPos,&posSize,m_resourceBndPos));

    int tableSize = ceil(m_simProperties.gridRes.x *m_simProperties.gridRes.y);
    checkCudaErrors(cudaMalloc(&m_fluidBuffers.bndCellOccBuff,tableSize*sizeof(int)));
    fillIntZero(m_cudaStream,m_threadsPerBlock,m_fluidBuffers.bndCellOccBuff,tableSize);
    checkCudaErrors(cudaMalloc(&m_fluidBuffers.bndCellIdxBuff,tableSize*sizeof(int)));
    fillIntZero(m_cudaStream,m_threadsPerBlock,m_fluidBuffers.bndCellIdxBuff,tableSize);

    hashAndSortBnd(m_cudaStream,m_threadsPerBlock,(int)bndTemp.size(),tableSize,m_fluidBuffers.bndPos,m_fluidBuffers.bndCellOccBuff,m_fluidBuffers.bndCellIdxBuff);

    checkCudaErrors(cudaGraphicsUnmapResources(1,&m_resourceBndPos));

    // Send these to the GPU
    updateGPUSimProps();

    // Create an OpenGL buffer for our position buffer
    // Create our VAO and vertex buffers
    glGenVertexArrays(1, &m_activeVAO);
    glBindVertexArray(m_activeVAO);

    // Put our vertices into an OpenGL buffer
    glGenBuffers(1, &m_posVBO);
    glBindBuffer(GL_ARRAY_BUFFER, m_posVBO);
    // We must alocate some space otherwise cuda cannot register it
    glBufferData(GL_ARRAY_BUFFER, sizeof(float3), NULL, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    // create our cuda graphics resource for our vertexs used for our OpenGL interop
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_resourcePos, m_posVBO, cudaGraphicsRegisterFlagsWriteDiscard));

    // Put our vertices into an OpenGL buffer
    glGenBuffers(1, &m_classVBO);
    glBindBuffer(GL_ARRAY_BUFFER, m_classVBO);
    // We must alocate some space otherwise cuda cannot register it
    glBufferData(GL_ARRAY_BUFFER, sizeof(float), NULL, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, 0);
    // create our cuda graphics resource for our vertexs used for our OpenGL interop
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_resourceClass, m_classVBO, cudaGraphicsRegisterFlagsWriteDiscard));

    // Unbind everything just in case
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}
//----------------------------------------------------------------------------------------------------------------------
SPHSolverCUDA::~SPHSolverCUDA()
{
    // Make sure we remember to unregister our cuda resource
    checkCudaErrors(cudaGraphicsUnregisterResource(m_resourcePos));
    checkCudaErrors(cudaGraphicsUnregisterResource(m_resourceClass));
    checkCudaErrors(cudaGraphicsUnregisterResource(m_resourceBndPos));

    // Delete our CUDA buffers
    if(m_fluidBuffers.velPtr) checkCudaErrors(cudaFree(m_fluidBuffers.velPtr));
    if(m_fluidBuffers.accPtr) checkCudaErrors(cudaFree(m_fluidBuffers.accPtr));
    if(m_fluidBuffers.cellIndexBuffer) checkCudaErrors(cudaFree(m_fluidBuffers.cellIndexBuffer));
    if(m_fluidBuffers.cellOccBuffer) checkCudaErrors(cudaFree(m_fluidBuffers.cellOccBuffer));
    if(m_fluidBuffers.hashKeys) checkCudaErrors(cudaFree(m_fluidBuffers.hashKeys));
    if(m_fluidBuffers.hashMap) checkCudaErrors(cudaFree(m_fluidBuffers.hashMap));
    if(m_fluidBuffers.denPtr) checkCudaErrors(cudaFree(m_fluidBuffers.denPtr));
    if(m_fluidBuffers.convergedPtr) checkCudaErrors(cudaFree(m_fluidBuffers.convergedPtr));
    if(m_fluidBuffers.bndCellIdxBuff) checkCudaErrors(cudaFree(m_fluidBuffers.bndCellIdxBuff));
    if(m_fluidBuffers.bndCellOccBuff) checkCudaErrors(cudaFree(m_fluidBuffers.bndCellOccBuff));
    if(m_fluidBuffers.pixelI) checkCudaErrors(cudaFree(m_fluidBuffers.pixelI));
    if(m_fluidBuffers.pixelCMYK) checkCudaErrors(cudaFree(m_fluidBuffers.pixelCMYK));
    // Make sure these are set to 0 just in case
    m_fluidBuffers.accPtr = 0;
    m_fluidBuffers.velPtr = 0;
    m_fluidBuffers.cellIndexBuffer = 0;
    m_fluidBuffers.cellOccBuffer = 0;
    m_fluidBuffers.hashKeys = 0;
    m_fluidBuffers.hashMap = 0;
    m_fluidBuffers.denPtr = 0;
    m_fluidBuffers.convergedPtr = 0;
    m_fluidBuffers.bndCellIdxBuff = 0;
    m_fluidBuffers.bndCellOccBuff = 0;
    m_fluidBuffers.pixelI = 0;
    m_fluidBuffers.pixelCMYK = 0;
    // Delete our CUDA streams as well
    checkCudaErrors(cudaStreamDestroy(m_cudaStream));
    // Delete our openGL objects
    glDeleteBuffers(1,&m_posVBO);
    glDeleteBuffers(1,&m_classVBO);
    glDeleteVertexArrays(1,&m_activeVAO);
    glDeleteBuffers(1,&m_bndVBO);
    glDeleteVertexArrays(1,&m_bndPosVAO);

}
//----------------------------------------------------------------------------------------------------------------------
void SPHSolverCUDA::setParticles(std::vector<float3> &_particles)
{
    // Set how many particles we have
    m_simProperties.numParticles = (int)_particles.size();

    // Unregister our resource
    checkCudaErrors(cudaGraphicsUnregisterResource(m_resourcePos));
    checkCudaErrors(cudaGraphicsUnregisterResource(m_resourceClass));

    // Fill our buffer with our positions
    glBindVertexArray(m_activeVAO);
    if(_particles.size())
    {
        glBindBuffer(GL_ARRAY_BUFFER, m_posVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(float3)*_particles.size(), &_particles[0], GL_DYNAMIC_DRAW);
        // create our cuda graphics resource for our vertexs used for our OpenGL interop
        checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_resourcePos, m_posVBO, cudaGraphicsRegisterFlagsWriteDiscard));

        // Generate some classes for our particles
        std::vector<float> classes;
        classes.resize(_particles.size());
        float ccount = 0.f;
        for(unsigned int i=0;i<classes.size();i++)
        {
            classes[i] = ccount;
            ccount+=1.f;
            if(ccount>3)ccount=0.f;
        }
        glBindBuffer(GL_ARRAY_BUFFER, m_classVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(float)*classes.size(), &classes[0], GL_DYNAMIC_DRAW);

        // create our cuda graphics resource for our vertexs used for our OpenGL interop
        checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_resourceClass, m_classVBO, cudaGraphicsRegisterFlagsWriteDiscard));


        // Delete our CUDA buffers fi they have anything in them
        if(m_fluidBuffers.velPtr) checkCudaErrors(cudaFree(m_fluidBuffers.velPtr));
        if(m_fluidBuffers.accPtr) checkCudaErrors(cudaFree(m_fluidBuffers.accPtr));
        if(m_fluidBuffers.denPtr) checkCudaErrors(cudaFree(m_fluidBuffers.denPtr));
        if(m_fluidBuffers.hashKeys) checkCudaErrors(cudaFree(m_fluidBuffers.hashKeys));
        if(m_fluidBuffers.convergedPtr) checkCudaErrors(cudaFree(m_fluidBuffers.convergedPtr));
        m_fluidBuffers.velPtr = 0;
        m_fluidBuffers.accPtr = 0;
        m_fluidBuffers.denPtr = 0;
        m_fluidBuffers.hashKeys = 0;
        m_fluidBuffers.convergedPtr = 0;
        // Fill them up with some blank data
        std::vector<float3> blankFloat3s;
        blankFloat3s.resize(_particles.size());
        for(unsigned int i=0;i<blankFloat3s.size();i++) blankFloat3s[i] = make_float3(0.f,0.f,0.f);

        // Send the data to the GPU
        checkCudaErrors(cudaMalloc(&m_fluidBuffers.velPtr,blankFloat3s.size()*sizeof(float3)));
        checkCudaErrors(cudaMemcpy(m_fluidBuffers.velPtr,&blankFloat3s[0],sizeof(float3)*blankFloat3s.size(),cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMalloc(&m_fluidBuffers.accPtr,blankFloat3s.size()*sizeof(float3)));
        checkCudaErrors(cudaMemcpy(m_fluidBuffers.accPtr,&blankFloat3s[0],sizeof(float3)*blankFloat3s.size(),cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMalloc(&m_fluidBuffers.hashKeys,_particles.size()*sizeof(int)));
        checkCudaErrors(cudaMalloc(&m_fluidBuffers.convergedPtr,_particles.size()*sizeof(int)));
        checkCudaErrors(cudaMalloc(&m_fluidBuffers.denPtr,_particles.size()*sizeof(float)));
        fillIntZero(m_cudaStream,m_threadsPerBlock,m_fluidBuffers.hashKeys,(int)_particles.size());
        fillIntZero(m_cudaStream,m_threadsPerBlock,m_fluidBuffers.convergedPtr,(int)_particles.size());

        //Send our sim properties to the GPU
        updateSimProps(&m_simProperties);

        // Map our pointer for our position data
        size_t posSize;
        checkCudaErrors(cudaGraphicsMapResources(1,&m_resourcePos));
        checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&m_fluidBuffers.posPtr,&posSize,m_resourcePos));
        checkCudaErrors(cudaGraphicsMapResources(1,&m_resourceClass));
        checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&m_fluidBuffers.classBuff,&posSize,m_resourceClass));

        int tableSize = (int)ceil(m_simProperties.gridRes.x *m_simProperties.gridRes.y);
        fillIntZero(m_cudaStream,m_threadsPerBlock,m_fluidBuffers.cellIndexBuffer,tableSize);
        fillIntZero(m_cudaStream,m_threadsPerBlock,m_fluidBuffers.cellOccBuffer,tableSize);

        // Hash and sort our particles
        hashAndSort(m_cudaStream, m_threadsPerBlock, m_simProperties.numParticles, tableSize , m_fluidBuffers);

        //unmap our buffer pointer and set it free into the wild
        checkCudaErrors(cudaGraphicsUnmapResources(1,&m_resourcePos));
        checkCudaErrors(cudaGraphicsUnmapResources(1,&m_resourceClass));

        //Set our volume if it hasnt already been set
        if(!m_volume)
        {
            m_volume = m_simProperties.mass * m_simProperties.numParticles;
        }
        else
        {
            m_simProperties.mass = m_volume/(m_simProperties.numParticles);
        }
    }
    else
    {
        glBindBuffer(GL_ARRAY_BUFFER, m_posVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(float3), NULL, GL_DYNAMIC_DRAW);

        glBindBuffer(GL_ARRAY_BUFFER, m_classVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(float), NULL, GL_DYNAMIC_DRAW);

        // create our cuda graphics resource for our vertexs used for our OpenGL interop
        checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_resourcePos, m_posVBO, cudaGraphicsRegisterFlagsWriteDiscard));        
        checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_resourcePos, m_classVBO, cudaGraphicsRegisterFlagsWriteDiscard));



        // Delete our CUDA buffers fi they have anything in them
        if(m_fluidBuffers.velPtr) checkCudaErrors(cudaFree(m_fluidBuffers.velPtr));
        if(m_fluidBuffers.accPtr) checkCudaErrors(cudaFree(m_fluidBuffers.accPtr));
        if(m_fluidBuffers.denPtr) checkCudaErrors(cudaFree(m_fluidBuffers.denPtr));
        if(m_fluidBuffers.hashKeys) checkCudaErrors(cudaFree(m_fluidBuffers.hashKeys));
        if(m_fluidBuffers.convergedPtr) checkCudaErrors(cudaFree(m_fluidBuffers.convergedPtr));
        m_fluidBuffers.velPtr = 0;
        m_fluidBuffers.accPtr = 0;
        m_fluidBuffers.denPtr = 0;
        m_fluidBuffers.hashKeys = 0;
        m_fluidBuffers.convergedPtr = 0;

    }

}
//----------------------------------------------------------------------------------------------------------------------
std::vector<float3> SPHSolverCUDA::getParticlePositions()
{
    // Map our pointer for our position data
    size_t posSize;
    checkCudaErrors(cudaGraphicsMapResources(1,&m_resourcePos));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&m_fluidBuffers.posPtr,&posSize,m_resourcePos));

    std::vector<float3> positions;
    positions.resize(m_simProperties.numParticles);

    // Copy our data from the GPU
    checkCudaErrors(cudaMemcpy(&positions[0],m_fluidBuffers.posPtr,sizeof(float3)*m_simProperties.numParticles,cudaMemcpyDeviceToHost));

    //unmap our buffer pointer and set it free into the wild
    checkCudaErrors(cudaGraphicsUnmapResources(1,&m_resourcePos));

    return positions;
}
//----------------------------------------------------------------------------------------------------------------------
std::vector<float2> SPHSolverCUDA::getParticlePosF2()
{
    std::vector<float3> posf3 = getParticlePositions();
    std::vector<float2> posf2;
    posf2.resize(posf3.size());
    for(unsigned int i=0;i<posf3.size();i++)
    {
        posf2[i] = make_float2(posf3[i].x,posf3[i].y);
    }
    return posf2;
}

//----------------------------------------------------------------------------------------------------------------------
void SPHSolverCUDA::genRandomSamples(float _n)
{
    float3 tempVec;
    boost::uniform_real<float> MinusPlusOneFloatDistrib(0.f, 1.f);
    boost::mt19937 rng(time(NULL));
    boost::variate_generator< boost::mt19937, boost::uniform_real<float> > gen(rng, MinusPlusOneFloatDistrib);
    std::vector<float3> positionsfloat;
    positionsfloat.resize(_n);
    for(unsigned int i=0;i<_n;i++)
    {
        tempVec = make_float3(gen(),gen(),0.f);
        while(tempVec.x==0.f || tempVec.x==1.f || tempVec.y==0.f || tempVec.y==1.f) tempVec = make_float3(gen(),gen(),0.f);
        tempVec.x*=m_simBounds.x;
        tempVec.y*=m_simBounds.y;
        positionsfloat[i] = make_float3(tempVec.x,tempVec.y,tempVec.z);
    }
    setParticles(positionsfloat);
}
//----------------------------------------------------------------------------------------------------------------------
void SPHSolverCUDA::setSmoothingLength(float _h)
{
    m_simProperties.h = _h;
    m_simProperties.hSqrd = _h*_h;
    m_simProperties.dWConst = 315.f/(64.f*(float)M_PI*_h*_h*_h*_h*_h*_h*_h*_h*_h);
    m_simProperties.pWConst = -45.f/((float)M_PI*_h*_h*_h*_h*_h*_h);
    m_simProperties.vWConst = 45.f/((float)M_PI*_h*_h*_h*_h*_h*_h);
    m_simProperties.cWConst1 = 32.f/((float)M_PI*_h*_h*_h*_h*_h*_h*_h*_h*_h);
    m_simProperties.cWConst2 = (_h*_h*_h*_h*_h*_h)/64.f;

    setHashPosAndDim(m_simProperties.gridMin,m_simProperties.gridDim);
}
//----------------------------------------------------------------------------------------------------------------------
void SPHSolverCUDA::setHashPosAndDim(float2 _gridMin, float2 _gridDim)
{
    m_simProperties.gridMin = _gridMin;
    m_simProperties.gridDim = _gridDim;
    m_simProperties.gridRes.x = (int)ceil(_gridDim.x/m_simProperties.h);
    m_simProperties.gridRes.y = (int)ceil(_gridDim.y/m_simProperties.h);
    int tableSize = ceil(m_simProperties.gridRes.x *m_simProperties.gridRes.y);

    // No point in alocating a buffer size of zero so lets just return
    if(tableSize==0)return;

    std::cout<<"table size "<<tableSize<<std::endl;

    // Remove anything that is in our bufferes currently
    if(m_fluidBuffers.cellIndexBuffer) checkCudaErrors(cudaFree(m_fluidBuffers.cellIndexBuffer));
    if(m_fluidBuffers.cellOccBuffer) checkCudaErrors(cudaFree(m_fluidBuffers.cellOccBuffer));
    if(m_fluidBuffers.hashMap) checkCudaErrors(cudaFree(m_fluidBuffers.hashMap));
    m_fluidBuffers.cellIndexBuffer = 0;
    m_fluidBuffers.cellOccBuffer = 0;
    m_fluidBuffers.hashMap = 0;
    // Send the data to our GPU buffers
    checkCudaErrors(cudaMalloc(&m_fluidBuffers.cellIndexBuffer,tableSize*sizeof(int)));
    checkCudaErrors(cudaMalloc(&m_fluidBuffers.cellOccBuffer,tableSize*sizeof(int)));
    // Fill with blank data
    fillIntZero(m_cudaStream,m_threadsPerBlock,m_fluidBuffers.cellOccBuffer,tableSize);

    // Update this our simulation properties on the GPU
    updateGPUSimProps();

    // Allocate memory for our hash map
    checkCudaErrors(cudaMalloc(&m_fluidBuffers.hashMap,tableSize*sizeof(cellInfo)));
    // Compute our hash map
    createHashMap(m_cudaStream,m_threadsPerBlock,tableSize,m_fluidBuffers);


}
//----------------------------------------------------------------------------------------------------------------------
void SPHSolverCUDA::update()
{
    //if no particles then theres no point in updating so just return
    if(!m_simProperties.numParticles)return;

    // Set our hash table values back to zero
    int tableSize = ceil(m_simProperties.gridRes.x *m_simProperties.gridRes.y);
    fillIntZero(m_cudaStream,m_threadsPerBlock,m_fluidBuffers.cellIndexBuffer,tableSize);
    fillIntZero(m_cudaStream,m_threadsPerBlock,m_fluidBuffers.cellOccBuffer,tableSize);

    //Send our sim properties to the GPU
    updateSimProps(&m_simProperties);

    // Map our pointer for our position data
    size_t posSize;
    checkCudaErrors(cudaGraphicsMapResources(1,&m_resourcePos));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&m_fluidBuffers.posPtr,&posSize,m_resourcePos));
    checkCudaErrors(cudaGraphicsMapResources(1,&m_resourceClass));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&m_fluidBuffers.classBuff,&posSize,m_resourceClass));
    checkCudaErrors(cudaGraphicsMapResources(1,&m_resourceBndPos));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&m_fluidBuffers.bndPos,&posSize,m_resourceBndPos));

    // Hash and sort our particles
    hashAndSort(m_cudaStream, m_threadsPerBlock, m_simProperties.numParticles, tableSize , m_fluidBuffers);

    // Compute our density
    initDensity(m_cudaStream,m_threadsPerBlock,m_simProperties.numParticles,m_fluidBuffers,m_multiclass);

    // Compute our rest density
    m_restDensity = getAverageDensity() - m_densityDiff;
    //std::cout<<"AvgDenstiy: "<<m_restDensity+m_densityDiff<<std::endl;

    // Solve for our new positions
    solve(m_cudaStream,m_threadsPerBlock,m_simProperties.numParticles,m_restDensity,m_fluidBuffers,m_multiclass);

    //unmap our buffer pointer and set it free into the wild
    checkCudaErrors(cudaGraphicsUnmapResources(1,&m_resourcePos));
    checkCudaErrors(cudaGraphicsUnmapResources(1,&m_resourceClass));
    checkCudaErrors(cudaGraphicsUnmapResources(1,&m_resourceBndPos));
}

//----------------------------------------------------------------------------------------------------------------------
void SPHSolverCUDA::setSampleImage(QString _loc)
{
    QImage img(_loc);
    QColor c;
    float i;
    std::vector<float> intensity;
    std::vector<float4> cmyk;
    intensity.resize(img.width()*img.height());
    cmyk.resize(img.width()*img.height());
    for(int x=0;x<img.width();x++)
    for(int y=0;y<img.width();y++)
    {
        c = QColor(img.pixel(x,y));
        i = 0.2989f*c.redF()+0.5870f*c.greenF()+0.1140f*c.blueF();
        intensity[x+(img.height()-1-y)*img.width()] = i;
        cmyk[i] = make_float4(c.cyanF(),c.magentaF(),c.yellowF(),c.blackF());
        std::cout<<cmyk[i].x<<","<<cmyk[i].y<<","<<cmyk[i].z<<","<<cmyk[i].w<<std::endl;
    }
    checkCudaErrors(cudaFree(m_fluidBuffers.pixelI));
    m_fluidBuffers.pixelI = 0;
    checkCudaErrors(cudaMalloc(&m_fluidBuffers.pixelI,sizeof(float)*intensity.size()));
    checkCudaErrors(cudaMemcpy(m_fluidBuffers.pixelI,&intensity[0],sizeof(float)*intensity.size(),cudaMemcpyHostToDevice));
    checkCudaErrors(cudaFree(m_fluidBuffers.pixelCMYK));
    m_fluidBuffers.pixelCMYK = 0;
    checkCudaErrors(cudaMalloc(&m_fluidBuffers.pixelCMYK,sizeof(float4)*cmyk.size()));
    checkCudaErrors(cudaMemcpy(m_fluidBuffers.pixelCMYK,&cmyk[0],sizeof(float4)*cmyk.size(),cudaMemcpyHostToDevice));

    updateGPUSimProps();
}
//----------------------------------------------------------------------------------------------------------------------
