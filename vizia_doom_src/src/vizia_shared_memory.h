#ifndef __VIZIA_SHARED_MEMORY_H__
#define __VIZIA_SHARED_MEMORY_H__

#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>

namespace bip = boost::interprocess;

extern bip::shared_memory_object viziaSM;

#define VIZIA_SM_NAME_BASE "ViziaSM"

void Vizia_SMInit(const char * id);

size_t Vizia_SMGetInputRegionBeginning();
size_t Vizia_SMGetGameVarsRegionBeginning();
size_t Vizia_SMGetScreenRegionBeginning();

void Vizia_SMClose();

#endif
