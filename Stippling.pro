# This specifies the exe name
TARGET=Stippling
# where to put the .o files
OBJECTS_DIR=obj
# core Qt Libs to use add more here if needed.
QT+=gui opengl core
# as I want to support 4.8 and 5 this will set a flag for some of the mac stuff
# mainly in the types.h file for the setMacVisual which is native in Qt5
isEqual(QT_MAJOR_VERSION, 5) {
	cache()
	DEFINES +=QT5BUILD
}
# where to put moc auto generated files
MOC_DIR=moc
# on a mac we don't create a .app bundle file ( for ease of multiplatform use)
CONFIG-=app_bundle
VPATH += ./src
# Auto include all .cpp files in the project src directory (can specifiy individually if required)
SOURCES+= src/main.cpp \
    src/MainWindow.cpp  \
    src/NGLScene.cpp  \
    src/Particle.cpp \
    src/ParticleDrawer.cpp \
    src/Shader.cpp \
    src/ShaderProgram.cpp \
    src/ShaderUtils.cpp \
    src/Camera.cpp \
    src/Text.cpp \
    src/SPHSolverCUDA.cpp

# same for the .h files
HEADERS+=include/MainWindow.h \
    include/NGLScene.h \
    include/Particle.h \
    include/ParticleDrawer.h \
    include/Shader.h \
    include/ShaderProgram.h \
    include/ShaderUtils.h \
    include/Camera.h \
    include/Text.h \
    include/SPHSolverCUDAKernals.h \
    include/SPHSolverCUDA.h

# and add the include dir into the search path for Qt and make
# Make sure you are not shadow building ot this will not work!
INCLUDEPATH +=./include
# where our exe is going to live (root of project)
DESTDIR=./
# add the glsl shader files
OTHER_FILES+= shaders/*.glsl \
                README.md
# were are going to default to a console app
CONFIG += console
DEFINES += _USE_MATH_DEFINES
#in on mac define DARWIN
macx:DEFINES+=DARWIN
win32:{
    DEFINES+=WIN32
    DEFINES+=_WIN32
    DEFINES += GLEW_STATIC
    INCLUDEPATH+=C:/boost
    LIBS+= -lopengl32 -lglew32s
}
# basic compiler flags (not all appropriate for all platforms)
QMAKE_CXXFLAGS+= -msse -msse2 -msse3
# use this to suppress some warning from boost
unix*:QMAKE_CXXFLAGS_WARN_ON += "-Wno-unused-parameter"


#----------------------------------------------------------------
#-------------------------Cuda setup-----------------------------
#----------------------------------------------------------------

#Enter your gencode here!
GENCODE = arch=compute_52,code=sm_52

#We must define this as we get some confilcs in minwindef.h and helper_math.h
DEFINES += NOMINMAX

#set out cuda sources
CUDA_SOURCES = "$$PWD"/cudaSrc/*.cu

#This is to add our .cu files to our file browser in Qt
SOURCES+=cudaSrc/*cu
SOURCES-=cudaSrc/*cu

# Path to cuda SDK install
macx:CUDA_DIR = /Developer/NVIDIA/CUDA-6.5
linux:CUDA_DIR = /usr/local/cuda-6.5
win32:CUDA_DIR = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5"
# Path to cuda toolkit install
macx:CUDA_SDK = /Developer/NVIDIA/CUDA-6.5/samples
linux:CUDA_SDK = /usr/local/cuda-6.5/samples
win32:CUDA_SDK = "C:\ProgramData\NVIDIA Corporation\CUDA Samples\v7.5"

#Cuda include paths
INCLUDEPATH += $$CUDA_DIR/include
#INCLUDEPATH += $$CUDA_DIR/common/inc/
#INCLUDEPATH += $$CUDA_DIR/../shared/inc/
#To get some prewritten helper functions from NVIDIA
win32:INCLUDEPATH += $$CUDA_SDK\common\inc


#cuda libs
macx:QMAKE_LIBDIR += $$CUDA_DIR/lib
linux:QMAKE_LIBDIR += $$CUDA_DIR/lib64
win32:QMAKE_LIBDIR += $$CUDA_DIR\lib\x64
linux|macx:QMAKE_LIBDIR += $$CUDA_SDK/common/lib
win32:QMAKE_LIBDIR +=$$CUDA_SDK\common\lib\x64
LIBS += -lcudart -lcudadevrt

# join the includes in a line
CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')

# nvcc flags (ptxas option verbose is always useful)
NVCCFLAGS = --compiler-options  -fno-strict-aliasing --ptxas-options=-v -maxrregcount 20 --use_fast_math

#On windows we must define if we are in debug mode or not
CONFIG(debug, debug|release) {
#DEBUG
    # MSVCRT link option (static or dynamic, it must be the same with your Qt SDK link option)
    win32:MSVCRT_LINK_FLAG_DEBUG = "/MDd"
    win32:NVCCFLAGS += -D_DEBUG -Xcompiler $$MSVCRT_LINK_FLAG_DEBUG
}
else{
#Release UNTESTED!!!
    win32:MSVCRT_LINK_FLAG_RELEASE = "/MD"
    win32:NVCCFLAGS += -Xcompiler $$MSVCRT_LINK_FLAG_RELEASE
}

#prepare intermediat cuda compiler
cudaIntr.input = CUDA_SOURCES
cudaIntr.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}.o
#So in windows object files have to be named with the .obj suffix instead of just .o
#God I hate you windows!!
win32:cudaIntr.output = $$OBJECTS_DIR/${QMAKE_FILE_BASE}.obj

## Tweak arch according to your hw's compute capability
cudaIntr.commands = $$CUDA_DIR/bin/nvcc -m64 -g -gencode $$GENCODE -dc $$NVCCFLAGS $$CUDA_INC $$LIBS ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}

#Set our variable out. These obj files need to be used to create the link obj file
#and used in our final gcc compilation
cudaIntr.variable_out = CUDA_OBJ
cudaIntr.variable_out += OBJECTS
cudaIntr.clean = cudaIntrObj/*.o
win32:cudaIntr.clean = cudaIntrObj/*.obj

QMAKE_EXTRA_UNIX_COMPILERS += cudaIntr


# Prepare the linking compiler step
cuda.input = CUDA_OBJ
cuda.output = ${QMAKE_FILE_BASE}_link.o
win32:cuda.output = ${QMAKE_FILE_BASE}_link.obj

# Tweak arch according to your hw's compute capability
cuda.commands = $$CUDA_DIR/bin/nvcc -m64 -g -gencode $$GENCODE  -dlink    ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
cuda.dependency_type = TYPE_C
cuda.depend_command = $$CUDA_DIR/bin/nvcc -g -M $$CUDA_INC $$NVCCFLAGS   ${QMAKE_FILE_NAME}
# Tell Qt that we want add more stuff to the Makefile
QMAKE_EXTRA_UNIX_COMPILERS += cuda

