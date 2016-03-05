#ifndef __VIZDOOM_SHARED_MEMORY_H__
#define __VIZDOOM_SHARED_MEMORY_H__

#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>

namespace bip = boost::interprocess;

extern bip::shared_memory_object vizdoomSM;

#define VIZDOOM_SM_NAME_BASE "ViZDoomSM"

void ViZDoom_SMInit(const char * id);
void ViZDoom_SMClose();

#endif
