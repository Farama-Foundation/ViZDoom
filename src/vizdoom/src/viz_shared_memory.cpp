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

#include "viz_shared_memory.h"
#include "viz_defines.h"
#include "viz_game.h"
#include "viz_input.h"
#include "viz_main.h"

#include "doomstat.h"
#include "v_video.h"

bip::shared_memory_object vizSM;
size_t vizSMSize;
char * vizSMName;

VIZSMRegion vizSMRegion[VIZ_SM_REGION_COUNT];

size_t vizSMGameStateOffset = 0;
size_t vizSMInputOffset = sizeof(VIZGameState);
size_t vizSMBuffersOffset = sizeof(VIZGameState) + sizeof(VIZInputState);


void VIZ_SMInit(const char * id){

    Printf("VIZ_SMInit: Init shared memory.\n");

    vizSMName = new char[strlen(VIZ_SM_NAME_BASE) + strlen(id) + 1];
    strcpy(vizSMName, VIZ_SM_NAME_BASE);
    strcat(vizSMName, id);

    try {
        bip::shared_memory_object::remove(vizSMName);
        vizSM = bip::shared_memory_object(bip::open_or_create, vizSMName, bip::read_write);

        vizSMSize = sizeof(VIZGameState) + sizeof(VIZInputState);
        vizSM.truncate(vizSMSize);

        VIZ_DebugMsg(1, VIZ_FUNC, "SMName: %s, SMSize: %zu", vizSMName, vizSMSize);
    }
    catch(...){ // bip::interprocess_exception
        VIZ_Error(VIZ_FUNC, "Failed to create shared memory.");
    }
}

void VIZ_SMUpdate(size_t buffersSize){
    try {
        vizSMSize = sizeof(VIZGameState) + sizeof(VIZInputState) + buffersSize;
        vizSM.truncate(vizSMSize);

        VIZ_DebugMsg(3, VIZ_FUNC, "New SMSize: %zu", vizSMSize);
    }
    catch(...){ // bip::interprocess_exception
        VIZ_Error(VIZ_FUNC, "Failed to truncate shared memory.");
    }
}

void VIZ_SMCreateRegion(VIZSMRegion* regionPtr, bool writeable, size_t offset, size_t size){
    regionPtr->offset = offset;
    regionPtr->size = size;
    regionPtr->writeable = writeable;
    if(regionPtr->size) {
        regionPtr->region = new bip::mapped_region(vizSM, bip::read_write, offset, size);
        regionPtr->address = regionPtr->region->get_address();
    }
}

void VIZ_SMDeleteRegion(VIZSMRegion* regionPtr) {
    if(regionPtr->region){
        delete regionPtr->region;
        regionPtr->region = NULL;
        regionPtr->address = NULL;
        regionPtr->size = 0;
        regionPtr->offset = 0;
        regionPtr->writeable = false;
    }
}

size_t VIZ_SMGetRegionOffset(VIZSMRegion* regionPtr){
    size_t offset = 0;
    for(auto i = &vizSMRegion[0]; i != regionPtr; ++i) offset += i->size;
    return offset;
}

void VIZ_SMClose(){
    for(int i = 0; i < VIZ_SM_REGION_COUNT; ++i) VIZ_SMDeleteRegion(&vizSMRegion[i]);

    //bip::shared_memory_object::remove(vizSMName);
	delete[] vizSMName;
}