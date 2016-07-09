#include "include/Particle.h"

int Particle::m_idMaker;

//----------------------------------------------------------------------------------------------------------------------
Particle::Particle(): m_pos (glm::vec3(0,0,0)), m_vel (glm::vec3(0,0,0)), m_acc (glm::vec3(0,0,0)), m_density (1.f), m_mass(1.f), m_id (++m_idMaker){}
//----------------------------------------------------------------------------------------------------------------------
Particle::Particle(glm::vec3 _pos, glm::vec3 _vel, glm::vec3 _acc) : m_pos (_pos), m_vel (_vel), m_acc (_acc), m_density (1.f), m_mass(1.f), m_id (++m_idMaker){}
//----------------------------------------------------------------------------------------------------------------------
std::ostream& operator<< (std::ostream& out, Particle& p){
   return out << "Pos ("<<p.getPos().x<<","<<p.getPos().y<<","<<p.getPos().z<<")"<< " Vel ("<<p.getVel().x<<","<<p.getVel().y<<","<<p.getVel().z<<")"<< " Acc ("<<p.getAcc().x<<","<<p.getAcc().y<<","<<p.getAcc().z<<") Density "<<p.getDensity() ;
}
//----------------------------------------------------------------------------------------------------------------------
