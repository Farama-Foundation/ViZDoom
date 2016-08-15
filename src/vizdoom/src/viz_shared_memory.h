/*
 Copyright (C) 2016 by Wojciech Jaśkowski, Michał Kempka, Grzegorz Runc, Jakub Toczek, Marek Wydmuch

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
*/

#ifndef __VIZ_SHARED_MEMORY_H__
#define __VIZ_SHARED_MEMORY_H__

#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>

#define VIZ_SM_NAME_BASE "ViZDoomSM"
#define VIZ_SM_REGION_COUNT 6

#define VIZ_SM_GAMESTATE    vizSMRegion[0]
#define VIZ_SM_INPUTSTATE   vizSMRegion[1]
#define VIZ_SM_SCREEN       vizSMRegion[2]
#define VIZ_SM_DEPTH        vizSMRegion[3]
#define VIZ_SM_LABELS       vizSMRegion[4]
#define VIZ_SM_AUTOMAP      vizSMRegion[5]

namespace bip = boost::interprocess;

extern bip::shared_memory_object vizSM;
extern size_t vizSMSize;

struct VIZSMRegion{
    bip::mapped_region *region;
    void *address;
    size_t offset;
    size_t size;
    bool writeable;

    VIZSMRegion(){
        this->region = NULL;
        this->address = NULL;
        this->offset = 0;
        this->size = 0;
        this->writeable = false;
    };
};

extern VIZSMRegion vizSMRegion[VIZ_SM_REGION_COUNT];

extern size_t vizSMGameStateOffset;
extern size_t vizSMInputOffset;
extern size_t vizSMBuffersOffset;

void VIZ_SMInit(const char * id);

void VIZ_SMUpdate(size_t buffersSize);

void VIZ_SMCreateRegion(VIZSMRegion* regionPtr, bool writeable, size_t offset, size_t size);

void VIZ_SMDeleteRegion(VIZSMRegion* regionPtr);

size_t VIZ_SMGetRegionOffset(VIZSMRegion* regionPtr);

void VIZ_SMClose();

#endif
