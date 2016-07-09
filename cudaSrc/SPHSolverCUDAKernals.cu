//----------------------------------------------------------------------------------------------------------------------
/// @file SPHSolverCUDAKernals.cu
/// @author Declan Russell
/// @date 03/02/2016
/// @version 1.0
//----------------------------------------------------------------------------------------------------------------------
#include "SPHSolverCUDAKernals.h"
#include <helper_math.h>  //< some math operations with cuda types
#include <iostream>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#define NULLHASH 4294967295
#define F_INVTWOPI  ( 0.15915494309f )
#define M_E ( 2.71828182845904523536f )

// Our simulation properties. These wont change much so lets load them into constant memory
__constant__ SimProps props;


//----------------------------------------------------------------------------------------------------------------------
__global__ void testKernal()
{
    printf("thread number %d\n",threadIdx.x);
}
//----------------------------------------------------------------------------------------------------------------------
__global__ void fillIntZeroKernal(int *_bufferPtr,int size)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx<size)
    {
        _bufferPtr[idx]=0;
    }
}
//----------------------------------------------------------------------------------------------------------------------
__global__ void createHashMapKernal(int _hashTableSize, fluidBuffers _buff)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx<_hashTableSize)
    {
        int count=0;
        int key;
        cellInfo cell;
        int y = floor((float)idx/(float)props.gridRes.x);
        int x = idx - (y*props.gridRes.x);
        int xi,yj;
        for(int i=-1;i<2;i++)
        {
            for(int j=-1;j<2;j++)
            {
                xi = x+i;
                yj = y+j;
                if ((xi>=0) && (xi<props.gridRes.x) && (yj>=0) && (yj<props.gridRes.y))
                {
                    key = idx + i + (j*props.gridRes.x);
                    if(key>=0 && key<_hashTableSize && count < _hashTableSize)
                    {
                        cell.cIdx[count] = key;
                        count++;
                    }
                }
            }
        }
        cell.cNum = count;
        _buff.hashMap[idx] = cell;
    }
}

//----------------------------------------------------------------------------------------------------------------------
__device__ int hashPos(float3 &_p)
{
    return floor((_p.x/props.gridDim.x)*props.gridRes.x) + (floor((_p.y/props.gridDim.y)*props.gridRes.y)*props.gridRes.x);
}
//----------------------------------------------------------------------------------------------------------------------
__global__ void hashParticles(int _numParticles,float3 *_posPtr, int *_hashKeys, int*_cellOccBuffer)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx<_numParticles)
    {
        float3 pos = _posPtr[idx];
        pos.x -= props.gridMin.x;
        pos.y -= props.gridMin.y;
        // Make sure the point is within our hash table
        if(pos.x>=0.f && pos.x<props.gridDim.x && pos.y>=0.f && pos.y<props.gridDim.y)
        {
            //Compute our hash key
            int key = hashPos(pos);
            _hashKeys[idx] = key;

            //Increment our occumpancy of this hash cell
            atomicAdd(&(_cellOccBuffer[key]), 1);

        }
        else
        {
            _hashKeys[idx] = NULLHASH;
            printf("NULL HASH idx %d pos %f,%f,%f\n",idx,pos.x,pos.y,pos.z);
        }
    }
}
//----------------------------------------------------------------------------------------------------------------------
__device__ float gausian(float3 _q, float3 _d, float _SDSqrd)
{
    float3 sq = (_q-_d);
    sq*=sq;
    float w = pow(M_E,-((sq.x+sq.y)/(2.f*_SDSqrd)));
    return w;
}
//----------------------------------------------------------------------------------------------------------------------
__device__ float calculatePressure(float &_pi,float &_restDensity)
{
    return props.k*(_pi-_restDensity);
}
//----------------------------------------------------------------------------------------------------------------------
__device__ float calcDensityWeighting(float _rLength)
{
    if(_rLength>0.f && _rLength<props.h)
    {
        return props.dWConst * (props.hSqrd - _rLength*_rLength) * (props.hSqrd - _rLength*_rLength) * (props.hSqrd - _rLength*_rLength);
    }
    else
    {
        return 0.f;
    }
}
//----------------------------------------------------------------------------------------------------------------------
__device__ float3 calcPressureWeighting(float3 &_r, float _rLength)
{
    if(_rLength>0.f && _rLength<props.h)
    {
        return props.pWConst * (_r) * (props.h - _rLength) * (props.h - _rLength);
    }
    else
    {
        return make_float3(0.f,0.f,0.f);
    }
}
//----------------------------------------------------------------------------------------------------------------------
__device__ float3 calcViscosityWeighting(float3 &_r, float &_rLength)
{
    if(_rLength>0.f && _rLength<=props.h)
    {
        return props.vWConst * _r * (props.h - _rLength);
    }
    else
    {
        return make_float3(0.f,0.f,0.f);
    }
}
//----------------------------------------------------------------------------------------------------------------------
__device__ float calcCoheWeighting(float _rLength)
{
    float w = 0.f;
    if(((2.f*_rLength)>props.h)&&(_rLength<=props.h))
    {
        w = props.cWConst1*((props.h-_rLength)*(props.h-_rLength)*(props.h-_rLength)*_rLength*_rLength*_rLength);
    }
    else if((_rLength>0.f)&&(2.f*_rLength<=props.h))
    {
        w = props.cWConst1*(2.f*((props.h-_rLength)*(props.h-_rLength)*(props.h-_rLength)*_rLength*_rLength*_rLength) - props.cWConst2);
    }
    return w;
}
//----------------------------------------------------------------------------------------------------------------------
__device__ float getPixelIntensity(float3 _p, float* _buff)
{
    float3 np = _p;
    np/=15.f;
    if(np.x<0.f || np.x>1.f || np.y<0.f || np.y>1.f)
    {
        printf("out of bounds\n");
        return 1.f;
    }
    np*=make_float3(199.f,199.f,0);
    np = floorf(np);
    return _buff[(int)np.x + (int)(np.y*200.f)];

}
//----------------------------------------------------------------------------------------------------------------------
__device__ float getPixelCMYK(float3 _p,float pClass, float4* _buff)
{
    float3 np = _p;
    np/=15.f;
    if(np.x<0.f || np.x>1.f || np.y<0.f || np.y>1.f)
    {
        printf("out of bounds\n");
        return 1.f;
    }
    np*=make_float3(199.f,199.f,0);
    np = floorf(np);
    float4 cmyk = _buff[(int)np.x + (int)(np.y*200.f)];
//    if(pClass==0.f) return cmyk.x;
//    if(pClass==1.f) return cmyk.y;
//    if(pClass==2.f) return cmyk.z;
//    if(pClass==3.f) return cmyk.w;
    return cmyk.x;

}
//----------------------------------------------------------------------------------------------------------------------
__device__ float invScale(float _var)
{
    return ((sqrt(_var))*0.999f)+0.001f;
}
//----------------------------------------------------------------------------------------------------------------------
__device__ float sizeFunction(float _rLength, float _vari, float _varj)
{
    float s = (2.f*_rLength)/(invScale(_vari)+invScale(_varj));
    if(s!=s)
    {
        printf("Nan isi %f isn %f, _rLength %f\n",_vari,_varj,_rLength);
        s = _rLength;
    }
//    printf("yi %f si %f yj %f sj %f s %f rLength %f\n",_pi.y,sizeFunc(_pi),_pn.y,sizeFunc(_pn),s,_rLength);

    return s;
}
//----------------------------------------------------------------------------------------------------------------------
__global__ void solveDensityMultiClassKernal(int _numParticles, fluidBuffers _buff)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx<_numParticles)
    {
        // Get our particle position
        float3 pi = _buff.posPtr[idx];
        int key = hashPos(pi-make_float3(props.gridMin.x,props.gridMin.y,0));
        // Get our neighbouring cell locations for this particle
        cellInfo nCells = _buff.hashMap[key];

        // Compute our density for all our particles
        int cellOcc;
        int cellIdx;
        int nIdx;
        float di = 0.f;
        float3 pj;
        float rLength,varj;
        float classI = _buff.classBuff[idx];
        float classJ,sf;
        float vari = getPixelCMYK(pi,classI,_buff.pixelCMYK);
        for(int c=0; c<nCells.cNum; c++)
        {
            // Get our cell occupancy total and start index
            cellOcc = _buff.cellOccBuffer[nCells.cIdx[c]];
            cellIdx = _buff.cellIndexBuffer[nCells.cIdx[c]];
            for(int i=0; i<cellOcc; i++)
            {
                //Get our neighbour particle index
                nIdx = cellIdx+i;
                //Dont want to compare against same particle
                if(nIdx==idx) continue;
                // Get our neighbour position
                pj = _buff.posPtr[nIdx];
                //Calculate our length
                rLength = length(pi-pj);
                classJ = _buff.classBuff[nIdx];
                varj = getPixelCMYK(pj,classJ,_buff.pixelCMYK);
                //Increment our density
                //di+=props.mass*calcDensityWeighting(rLength);
                sf = sizeFunction(rLength,vari,varj);
                //if(classI!=classJ) sf*=3.f;
                di+=props.mass*calcDensityWeighting(sf);
            }
            // Do the same thing but for our boundary ghost particles
            // Get our cell occupancy total and start index
            cellOcc = _buff.bndCellOccBuff[nCells.cIdx[c]];
            cellIdx = _buff.bndCellIdxBuff[nCells.cIdx[c]];
            for(int i=0; i<cellOcc; i++)
            {
                //Get our neighbour particle index
                nIdx = cellIdx+i;
                // Get our neighbour position
                pj = _buff.bndPos[nIdx];
                //Calculate our length
                rLength = length(pi-pj);
                //Increment our density
                //di+=props.mass*calcDensityWeighting(rLength);
                di+=props.mass*calcDensityWeighting(sizeFunction(rLength,vari,1.f));
            }
        }
        _buff.denPtr[idx] = di;
    }
}
//----------------------------------------------------------------------------------------------------------------------
__global__ void solveDensityKernal(int _numParticles, fluidBuffers _buff)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx<_numParticles)
    {
        // Get our particle position
        float3 pi = _buff.posPtr[idx];
        int key = hashPos(pi-make_float3(props.gridMin.x,props.gridMin.y,0));
        // Get our neighbouring cell locations for this particle
        cellInfo nCells = _buff.hashMap[key];

        // Compute our density for all our particles
        int cellOcc;
        int cellIdx;
        int nIdx;
        float di = 0.f;
        float3 pj;
        float vari = getPixelIntensity(pi,_buff.pixelI);
        float rLength,varj;
        for(int c=0; c<nCells.cNum; c++)
        {
            // Get our cell occupancy total and start index
            cellOcc = _buff.cellOccBuffer[nCells.cIdx[c]];
            cellIdx = _buff.cellIndexBuffer[nCells.cIdx[c]];
            for(int i=0; i<cellOcc; i++)
            {
                //Get our neighbour particle index
                nIdx = cellIdx+i;
                //Dont want to compare against same particle
                if(nIdx==idx) continue;
                // Get our neighbour position
                pj = _buff.posPtr[nIdx];
                //Calculate our length
                rLength = length(pi-pj);
                varj = getPixelIntensity(pj,_buff.pixelI);
                //Increment our density
                //di+=props.mass*calcDensityWeighting(rLength);
                di+=props.mass*calcDensityWeighting(sizeFunction(rLength,vari,varj));
            }
            // Do the same thing but for our boundary ghost particles
            // Get our cell occupancy total and start index
            cellOcc = _buff.bndCellOccBuff[nCells.cIdx[c]];
            cellIdx = _buff.bndCellIdxBuff[nCells.cIdx[c]];
            for(int i=0; i<cellOcc; i++)
            {
                //Get our neighbour particle index
                nIdx = cellIdx+i;
                // Get our neighbour position
                pj = _buff.bndPos[nIdx];
                //Calculate our length
                rLength = length(pi-pj);
                //Increment our density
                //di+=props.mass*calcDensityWeighting(rLength);
                di+=props.mass*calcDensityWeighting(sizeFunction(rLength,vari,1.f));
            }
        }
        _buff.denPtr[idx] = di;
    }
}
//----------------------------------------------------------------------------------------------------------------------
__global__ void solveForcesMultiClassKernal(int _numParticles, float _restDensity, fluidBuffers _buff)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx<_numParticles)
    {
        // Get our particle position
        float3 pi = _buff.posPtr[idx];
        float di = _buff.denPtr[idx];
        float3 acc = make_float3(0.f,0.f,0.f);
        float avgLen = 0.f;
        // Put this in its own scope means we get some registers back at the end of it (I think)
        if(di>0.f)
        {
            // Get our neighbouring cell locations for this particle
            cellInfo nCells = _buff.hashMap[hashPos(pi-make_float3(props.gridMin.x,props.gridMin.y,0))];

            // Compute our fources for all our particles
            int cellOcc,cellIdx,nIdx;
            float classI = _buff.classBuff[idx];
            float classJ,sf;
            float vari = getPixelCMYK(pi,classI,_buff.pixelCMYK);
            float dj,presi,presj,rLength,varj;
            presi = calculatePressure(di,_restDensity);
            int numN = 0;
            float3 pj,r,w;
            float3 presForce = make_float3(0.f,0.f,0.f);
            float3 coheForce = make_float3(0.f,0.f,0.f);
            for(int c=0; c<nCells.cNum; c++)
            {
                // Get our cell occupancy total and start index
                cellOcc = _buff.cellOccBuffer[nCells.cIdx[c]];
                cellIdx = _buff.cellIndexBuffer[nCells.cIdx[c]];
                for(int i=0; i<cellOcc; i++)
                {
                    //Get our neighbour particle index
                    nIdx = cellIdx+i;
                    //Dont want to compare against same particle
                    if(nIdx==idx) continue;
                    // Get our neighbour density
                    dj = _buff.denPtr[nIdx];
                    if(dj>0.f)
                    {
                        // Get our neighbour position
                        pj = _buff.posPtr[nIdx];
                        //Get our vector beteen points
                        r = pi - pj;
                        //Calculate our length
                        rLength=length(r);
                        // Normalise our differential
                        r/=rLength;

                        //Compute our particles pressure
                        presj = calculatePressure(dj,_restDensity);
                        classJ = _buff.classBuff[nIdx];
                        varj = getPixelCMYK(pj,classJ,_buff.pixelCMYK);
                        //Weighting
                        //w = calcPressureWeighting(r,rLength);
                        sf = sizeFunction(rLength,vari,varj);
                        //if(classI!=classJ) sf*=3.f;
                        w = calcPressureWeighting(r,sf);
                        // Accumilate our pressure force
                        presForce+= ((presi/(di*di)) + (presj/(dj*dj))) * props.mass * w;
                        // Accumilate our cohesion force
//                        cw = calcCoheWeighting(sizeFunction(rLength,vari,varj));
//                        if(cw!=cw) printf("cw %f\n",cw);
//                        coheForce+=-props.tension*props.mass*props.mass*((2.f*_restDensity)/(di+dj))*r*cw;

                        avgLen+=rLength;
                        numN++;
                    }

                }
                // Do the same for our boundary ghost particles
                // Get our cell occupancy total and start index
                cellOcc = _buff.bndCellOccBuff[nCells.cIdx[c]];
                cellIdx = _buff.bndCellIdxBuff[nCells.cIdx[c]];
                for(int i=0; i<cellOcc; i++)
                {
                    //Get our neighbour particle index
                    nIdx = cellIdx+i;
                    // Get our neighbour position
                    pj = _buff.bndPos[nIdx];
                    //Get our vector beteen points
                    r = pi - pj;
                    //Calculate our length
                    rLength=length(r);
                    // Normalise our differential
                    r/=rLength;

                    //Weighting
                    w = calcPressureWeighting(r,sizeFunction(rLength,vari,1.f));

                    // Accumilate our pressure force
                    presForce+= (presi/(di*di)) * props.mass * w;

                    // Accumilate our cohesion force
//                    cw = calcCoheWeighting(rLength);
//                    if(cw!=cw) printf("cw %f\n",cw);
//                    coheForce+=-props.tension*props.mass*props.mass*((2.f*_restDensity)/(di+_restDensity))*r*cw;
                }
            }

            // Compute our average distance between neighbours
            avgLen/=numN;
            // Complete our pressure force term
            presForce*=-1.f*props.mass;
            acc = (presForce+coheForce)/props.mass;
        }

        // Acceleration limit
//        if(dot(acc,acc)>props.accLimit2)
//        {
//            acc *= props.accLimit/length(acc);
//        }

        // Now lets integerate our acceleration using leapfrog to get our new position
        float3 halfBwd = _buff.velPtr[idx] - 0.5f*props.timeStep*acc;
        float3 halfFwd = halfBwd + props.timeStep*acc;
        // Apply velocity dampaning
        halfFwd *= 0.9f;

        //printf("vel %f,%f,%f\n",halfFwd.x,halfFwd.y,halfFwd.z);

        //Velocity Limit
//        if(dot(halfFwd,halfFwd)>props.velLimit2)
//        {
//            halfFwd *= props.velLimit/length(halfFwd);
//        }

        // Update our velocity
        _buff.velPtr[idx] = halfFwd;


        // Update our position
        float3 oldPos = pi;
        pi+= props.timeStep * halfFwd;


        //Place our particles back in our bounds
        //this could potentially have problems with velocity and such not being
        //adjusted but we will leave that for future work (Hes says...)
        if(pi.x<0.f){
            pi.x = 0.f;
        }
        if(pi.y<0.f){
            pi.y = 0.f;
        }
        if(pi.x>15.f){
            pi.x = 15.f;
        }
        if(pi.y>15.f){
            pi.y = 15.f;
        }

        _buff.posPtr[idx] = pi;
        // Check to see if we have met our converged state
        if(length(oldPos-pi)<props.convergeValue*avgLen)
        {
            _buff.convergedPtr[idx] = 1;
        }
        else
        {
            _buff.convergedPtr[idx] = 0;
        }
    }
}
//----------------------------------------------------------------------------------------------------------------------
__global__ void solveForcesKernal(int _numParticles, float _restDensity, fluidBuffers _buff)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx<_numParticles)
    {
        // Get our particle position
        float3 pi = _buff.posPtr[idx];
        float di = _buff.denPtr[idx];
        float3 acc = make_float3(0.f,0.f,0.f);
        float avgLen = 0.f;
        // Put this in its own scope means we get some registers back at the end of it (I think)
        if(di>0.f)
        {
            // Get our neighbouring cell locations for this particle
            cellInfo nCells = _buff.hashMap[hashPos(pi-make_float3(props.gridMin.x,props.gridMin.y,0))];

            // Compute our fources for all our particles
            int cellOcc,cellIdx,nIdx;
            float vari = getPixelIntensity(pi,_buff.pixelI);
            float dj,presi,presj,rLength,varj;
            presi = calculatePressure(di,_restDensity);
            int numN = 0;
            float3 pj,r,w;
            float3 presForce = make_float3(0.f,0.f,0.f);
            float3 coheForce = make_float3(0.f,0.f,0.f);
            for(int c=0; c<nCells.cNum; c++)
            {
                // Get our cell occupancy total and start index
                cellOcc = _buff.cellOccBuffer[nCells.cIdx[c]];
                cellIdx = _buff.cellIndexBuffer[nCells.cIdx[c]];
                for(int i=0; i<cellOcc; i++)
                {
                    //Get our neighbour particle index
                    nIdx = cellIdx+i;
                    //Dont want to compare against same particle
                    if(nIdx==idx) continue;
                    // Get our neighbour density
                    dj = _buff.denPtr[nIdx];
                    if(dj>0.f)
                    {
                        // Get our neighbour position
                        pj = _buff.posPtr[nIdx];
                        //Get our vector beteen points
                        r = pi - pj;
                        //Calculate our length
                        rLength=length(r);
                        // Normalise our differential
                        r/=rLength;

                        //Compute our particles pressure
                        presj = calculatePressure(dj,_restDensity);

                        varj = getPixelIntensity(pj,_buff.pixelI);
                        //Weighting
                        //w = calcPressureWeighting(r,rLength);
                        w = calcPressureWeighting(r,sizeFunction(rLength,vari,varj));
                        // Accumilate our pressure force
                        presForce+= ((presi/(di*di)) + (presj/(dj*dj))) * props.mass * w;
                        // Accumilate our cohesion force
//                        cw = calcCoheWeighting(sizeFunction(rLength,vari,varj));
//                        if(cw!=cw) printf("cw %f\n",cw);
//                        coheForce+=-props.tension*props.mass*props.mass*((2.f*_restDensity)/(di+dj))*r*cw;

                        avgLen+=rLength;
                        numN++;
                    }

                }
                // Do the same for our boundary ghost particles
                // Get our cell occupancy total and start index
                cellOcc = _buff.bndCellOccBuff[nCells.cIdx[c]];
                cellIdx = _buff.bndCellIdxBuff[nCells.cIdx[c]];
                for(int i=0; i<cellOcc; i++)
                {
                    //Get our neighbour particle index
                    nIdx = cellIdx+i;
                    // Get our neighbour position
                    pj = _buff.bndPos[nIdx];
                    //Get our vector beteen points
                    r = pi - pj;
                    //Calculate our length
                    rLength=length(r);
                    // Normalise our differential
                    r/=rLength;

                    //Weighting
                    w = calcPressureWeighting(r,sizeFunction(rLength,vari,1.f));

                    // Accumilate our pressure force
                    presForce+= (presi/(di*di)) * props.mass * w;

                    // Accumilate our cohesion force
//                    cw = calcCoheWeighting(rLength);
//                    if(cw!=cw) printf("cw %f\n",cw);
//                    coheForce+=-props.tension*props.mass*props.mass*((2.f*_restDensity)/(di+_restDensity))*r*cw;
                }
            }

            // Compute our average distance between neighbours
            avgLen/=numN;
            // Complete our pressure force term
            presForce*=-1.f*props.mass;
            acc = (presForce+coheForce)/props.mass;
        }

        // Acceleration limit
//        if(dot(acc,acc)>props.accLimit2)
//        {
//            acc *= props.accLimit/length(acc);
//        }

        // Now lets integerate our acceleration using leapfrog to get our new position
        float3 halfBwd = _buff.velPtr[idx] - 0.5f*props.timeStep*acc;
        float3 halfFwd = halfBwd + props.timeStep*acc;
        // Apply velocity dampaning
        halfFwd *= 0.9f;

        //printf("vel %f,%f,%f\n",halfFwd.x,halfFwd.y,halfFwd.z);

        //Velocity Limit
//        if(dot(halfFwd,halfFwd)>props.velLimit2)
//        {
//            halfFwd *= props.velLimit/length(halfFwd);
//        }

        // Update our velocity
        _buff.velPtr[idx] = halfFwd;


        // Update our position
        float3 oldPos = pi;
        pi+= props.timeStep * halfFwd;


        //Place our particles back in our bounds
        //this could potentially have problems with velocity and such not being
        //adjusted but we will leave that for future work (Hes says...)
        if(pi.x<0.f){
            pi.x = 0.f;
        }
        if(pi.y<0.f){
            pi.y = 0.f;
        }
        if(pi.x>15.f){
            pi.x = 15.f;
        }
        if(pi.y>15.f){
            pi.y = 15.f;
        }

        _buff.posPtr[idx] = pi;
        // Check to see if we have met our converged state
        if(length(oldPos-pi)<props.convergeValue*avgLen)
        {
            _buff.convergedPtr[idx] = 1;
        }
        else
        {
            _buff.convergedPtr[idx] = 0;
        }
    }
}
//----------------------------------------------------------------------------------------------------------------------
void test(){
    printf("calling\n");
    testKernal<<<1,1000>>>();
    //make sure all our threads are done
    cudaThreadSynchronize();
    printf("called\n");
}
//----------------------------------------------------------------------------------------------------------------------
float computeAverageDensity(int _numParticles, fluidBuffers _buff)
{
    // Turn our density buffer pointer into a thrust iterater
    thrust::device_ptr<float> t_denPtr = thrust::device_pointer_cast(_buff.denPtr);

    // Use reduce to sum all our densities
    float sum = thrust::reduce(t_denPtr, t_denPtr+_numParticles, 0.f, thrust::plus<float>());

    // Return our average density
    return sum/(float)_numParticles;
}
//----------------------------------------------------------------------------------------------------------------------
void updateSimProps(SimProps *_props)
{
    #ifdef CUDA_42
        // Unlikely we will ever use CUDA 4.2 but nice to have it in anyway I guess?
        cudaMemcpyToSymbol ( "props", _props, sizeof(SimProps) );
    #else
        cudaMemcpyToSymbol ( props, _props, sizeof(SimProps) );
    #endif
}
//----------------------------------------------------------------------------------------------------------------------
void fillIntZero(cudaStream_t _stream, int _threadsPerBlock, int *_bufferPtr,int size)
{
    if(size>_threadsPerBlock)
    {
        //calculate how many blocks we want
        int blocks = ceil(size/_threadsPerBlock)+1;
        fillIntZeroKernal<<<blocks,_threadsPerBlock,0,_stream>>>(_bufferPtr,size);
    }
    else{
        fillIntZeroKernal<<<1,size,0,_stream>>>(_bufferPtr,size);
    }
    //make sure all our threads are done
    cudaThreadSynchronize();
    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
      // print the CUDA error message and exit
      printf("Fill int zero: %s\n", cudaGetErrorString(error));
      exit(-1);
    }
}
//----------------------------------------------------------------------------------------------------------------------
void createHashMap(cudaStream_t _stream, int _threadsPerBlock, int _hashTableSize, fluidBuffers _buff)
{
    int blocks = 1;
    int threads = _hashTableSize;
    if(_hashTableSize>_threadsPerBlock)
    {
        //calculate how many blocks we want
        blocks = ceil(_hashTableSize/_threadsPerBlock)+1;
        threads = _threadsPerBlock;
    }

    // Create ou hash map
    createHashMapKernal<<<blocks,threads,0,_stream>>>(_hashTableSize,_buff);

    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
      // print the CUDA error message and exit
      printf("Create hash map error: %s\n", cudaGetErrorString(error));
      exit(-1);
    }

    //make sure all our threads are done
    cudaThreadSynchronize();
}
//----------------------------------------------------------------------------------------------------------------------
void hashAndSortBnd(cudaStream_t _stream, int _threadsPerBlock, int _numParticles, int _hashTableSize, float3 *posPtr, int *_occPtr, int *_idxPtr)
{
    int blocks = 1;
    int threads = _numParticles;
    if(_numParticles>_threadsPerBlock)
    {
        //calculate how many blocks we want
        blocks = ceil(_numParticles/_threadsPerBlock)+1;
        threads = _threadsPerBlock;
    }

    int *hashKeys = 0;
    cudaMalloc(&hashKeys,_numParticles*sizeof(int));
    fillIntZero(_stream,_threadsPerBlock,hashKeys,_numParticles);

    //Hash our partilces
    hashParticles<<<blocks,threads,0,_stream>>>(_numParticles,posPtr,hashKeys,_occPtr);

    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
      // print the CUDA error message and exit
      printf("Hash Particles error: %s\n", cudaGetErrorString(error));
      exit(-1);
    }

    //make sure all our threads are done
    cudaThreadSynchronize();

//    //Turn our raw pointers into thrust pointers so we can use
//    //thrusts sort algorithm
    thrust::device_ptr<int> t_hashPtr = thrust::device_pointer_cast(hashKeys);
    thrust::device_ptr<float3> t_posPtr = thrust::device_pointer_cast(posPtr);
    thrust::device_ptr<int> t_cellOccPtr = thrust::device_pointer_cast(_occPtr);
    thrust::device_ptr<int> t_cellIdxPtr = thrust::device_pointer_cast(_idxPtr);

    //sort our buffers
    thrust::sort_by_key(t_hashPtr,t_hashPtr+_numParticles, t_posPtr);
    //make sure all our threads are done
    cudaThreadSynchronize();


    //Create our cell indexs
    //run an excludive scan on our arrays to do this
    thrust::exclusive_scan(t_cellOccPtr,t_cellOccPtr+_hashTableSize,t_cellIdxPtr);

    //make sure all our threads are done
    cudaThreadSynchronize();

    cudaFree(hashKeys);

    //DEBUG: uncomment to print out counted cell occupancy
    //thrust::copy(t_cellOccPtr, t_cellOccPtr+_hashTableSize, std::ostream_iterator<unsigned int>(std::cout, " "));
    //std::cout<<"\n"<<std::endl;
}
//----------------------------------------------------------------------------------------------------------------------
void hashAndSort(cudaStream_t _stream,int _threadsPerBlock, int _numParticles, int _hashTableSize, fluidBuffers _buff)
{
    int blocks = 1;
    int threads = _numParticles;
    if(_numParticles>_threadsPerBlock)
    {
        //calculate how many blocks we want
        blocks = ceil(_numParticles/_threadsPerBlock)+1;
        threads = _threadsPerBlock;
    }

    //Hash our partilces
    hashParticles<<<blocks,threads,0,_stream>>>(_numParticles,_buff.posPtr,_buff.hashKeys,_buff.cellOccBuffer);

    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
      // print the CUDA error message and exit
      printf("Hash Particles error: %s\n", cudaGetErrorString(error));
      exit(-1);
    }

    //make sure all our threads are done
    cudaThreadSynchronize();

//    //Turn our raw pointers into thrust pointers so we can use
//    //thrusts sort algorithm
    thrust::device_ptr<int> t_hashPtr = thrust::device_pointer_cast(_buff.hashKeys);
    thrust::device_ptr<float3> t_posPtr = thrust::device_pointer_cast(_buff.posPtr);
    thrust::device_ptr<float3> t_velPtr = thrust::device_pointer_cast(_buff.velPtr);
    thrust::device_ptr<float3> t_accPtr = thrust::device_pointer_cast(_buff.accPtr);
    thrust::device_ptr<float> t_classPtr = thrust::device_pointer_cast(_buff.classBuff);
    thrust::device_ptr<int> t_cellOccPtr = thrust::device_pointer_cast(_buff.cellOccBuffer);
    thrust::device_ptr<int> t_cellIdxPtr = thrust::device_pointer_cast(_buff.cellIndexBuffer);

    //sort our buffers
    thrust::sort_by_key(t_hashPtr,t_hashPtr+_numParticles, thrust::make_zip_iterator(thrust::make_tuple(t_posPtr,t_velPtr,t_accPtr,t_classPtr)));
    //make sure all our threads are done
    cudaThreadSynchronize();


    //Create our cell indexs
    //run an excludive scan on our arrays to do this
    thrust::exclusive_scan(t_cellOccPtr,t_cellOccPtr+_hashTableSize,t_cellIdxPtr);

    //make sure all our threads are done
    cudaThreadSynchronize();

    //DEBUG: uncomment to print out counted cell occupancy
    //thrust::copy(t_cellOccPtr, t_cellOccPtr+_hashTableSize, std::ostream_iterator<unsigned int>(std::cout, " "));
    //std::cout<<"\n"<<std::endl;
    //thrust::copy(t_classPtr, t_classPtr+_numParticles, std::ostream_iterator<int>(std::cout, " "));
    //std::cout<<"\n"<<std::endl;
}
//----------------------------------------------------------------------------------------------------------------------
void initDensity(cudaStream_t _stream, int _threadsPerBlock, int _numParticles, fluidBuffers _buff, bool _multiClass)
{
    int blocks = 1;
    int threads = _numParticles;
    if(_numParticles>_threadsPerBlock)
    {
        //calculate how many blocks we want
        blocks = ceil(_numParticles/_threadsPerBlock)+1;
        threads = _threadsPerBlock;
    }

    //Solve our particles density
    if(_multiClass)
    {
        solveDensityMultiClassKernal<<<blocks,threads,0,_stream>>>(_numParticles,_buff);
    }
    else
    {
        solveDensityKernal<<<blocks,threads,0,_stream>>>(_numParticles,_buff);
    }
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
      // print the CUDA error message and exit
      printf("Solve Density Kernel error: %s\n", cudaGetErrorString(error));
      exit(-1);
    }

    //make sure all our threads are done
    cudaThreadSynchronize();

    //DEBUG: uncomment to print out counted density
    // Turn our density buffer pointer into a thrust iterater
    //thrust::device_ptr<float> t_denPtr = thrust::device_pointer_cast(_buff.denPtr);
    //thrust::copy(t_denPtr, t_denPtr+_numParticles, std::ostream_iterator<unsigned int>(std::cout, " "));
    //std::cout<<"\n"<<std::endl;
}
//----------------------------------------------------------------------------------------------------------------------
void solve(cudaStream_t _stream, int _threadsPerBlock, int _numParticles, float _restDensity, fluidBuffers _buff, bool _multiClass)
{
    int blocks = 1;
    int threads = _numParticles;
    if(_numParticles>_threadsPerBlock)
    {
        //calculate how many blocks we want
        blocks = ceil(_numParticles/_threadsPerBlock)+1;
        threads = _threadsPerBlock;
    }


    //Solve for our new positions
    if(_multiClass)
    {
        solveForcesKernal<<<blocks,threads,0,_stream>>>(_numParticles, _restDensity, _buff);
    }
    else
    {
        solveForcesKernal<<<blocks,threads,0,_stream>>>(_numParticles, _restDensity, _buff);
    }

    //make sure all our threads are done
    cudaThreadSynchronize();

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
      // print the CUDA error message and exit
      printf("Solve error: %s\n", cudaGetErrorString(error));
      exit(-1);
    }

}
//----------------------------------------------------------------------------------------------------------------------
bool isConverged(int _numParticles, fluidBuffers _buff)
{
    // Turn our density buffer pointer into a thrust iterater
    thrust::device_ptr<int> t_conPtr = thrust::device_pointer_cast(_buff.convergedPtr);

    // Use reduce to sum all our densities
    int sum = thrust::reduce(t_conPtr, t_conPtr+_numParticles, 0, thrust::plus<int>());

    return (sum==_numParticles);
}
//----------------------------------------------------------------------------------------------------------------------
