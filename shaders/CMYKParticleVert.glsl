//----------------------------------------------------------------------------------------------------------------------
/// @file ParticleVert.glsl
/// @author Declan Russell
/// @date 17/11/15
/// @version 1.0
/// @namepsace GLSL
/// @class ParticleVert
/// @brief Vertex shader for our particle shader. Takes in points of particles and scales the point sprite with
/// @brief the current scene projection matrix.
//----------------------------------------------------------------------------------------------------------------------

#version 400

//----------------------------------------------------------------------------------------------------------------------
/// @brief position of particles buffer
//----------------------------------------------------------------------------------------------------------------------
layout (location = 0) in vec3 vertexPosition;
//----------------------------------------------------------------------------------------------------------------------
/// @brief class of particles buffer
//----------------------------------------------------------------------------------------------------------------------
layout (location = 1) in float vertexClass;

//----------------------------------------------------------------------------------------------------------------------
/// @brief eye space position to be sent to fragment shader
//----------------------------------------------------------------------------------------------------------------------
out vec3 position;
out vec3 colour;
//----------------------------------------------------------------------------------------------------------------------
/// @brief our projection matrix
//----------------------------------------------------------------------------------------------------------------------
uniform mat4 P;
//----------------------------------------------------------------------------------------------------------------------
/// @brief our model view matrix
//----------------------------------------------------------------------------------------------------------------------
uniform mat4 MV;
//----------------------------------------------------------------------------------------------------------------------
/// @brief our model view projection matrix
//----------------------------------------------------------------------------------------------------------------------
uniform mat4 MVP;
//----------------------------------------------------------------------------------------------------------------------
/// @brief the width of our screen
//----------------------------------------------------------------------------------------------------------------------
uniform int screenWidth;
//----------------------------------------------------------------------------------------------------------------------
/// @breif the size of our particles
//----------------------------------------------------------------------------------------------------------------------
uniform float pointSize;

out vec3 ogPos;

float sizeFunc(vec3 _p)
{
    return ((abs(_p.y))*0.5f)+0.5f;
}
//----------------------------------------------------------------------------------------------------------------------
/// @brief vertex main. Scales point sprites with projection matrix and passes through eye space postion to fragment shader
//----------------------------------------------------------------------------------------------------------------------
void main(){
    //vec3 pos = vertexPosition * sizeFunc(vertexPosition);
    //scale the point sprite based on our projection matrix
    //vec4 eyePos = MV * vec4(pos,1.0);
    if(vertexClass==0){ colour = vec3(0,1,1);}
    if(vertexClass==1){ colour = vec3(1,0,1);}
    if(vertexClass==2){ colour = vec3(1,1,0);}
    if(vertexClass>2){ colour = vec3(vertexClass);}

    ogPos = vertexPosition;
    vec4 eyePos = MV * vec4(vertexPosition,1.0);
    position = vec3(eyePos);
    vec4 projCorner = P * vec4(0.5*pointSize, 0.5*pointSize, eyePos.z, eyePos.w);
    gl_PointSize = screenWidth * projCorner.x / projCorner.w;
    //gl_Position = vec4(MVP * vec4(pos, 1.0));
    gl_Position = vec4(MVP * vec4(vertexPosition, 1.0));
}
