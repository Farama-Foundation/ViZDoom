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

#include "viz_screen.h"
#include "viz_defines.h"
#include "viz_shared_memory.h"
#include "viz_message_queue.h"
#include "viz_depth.h"
#include "viz_labels.h"

#include "d_main.h"
#include "doomstat.h"
#include "v_video.h"

unsigned int vizScreenHeight;
unsigned int vizScreenWidth;
size_t vizScreenPitch;
size_t vizScreenSize;
size_t vizScreenChannelSize;

int posMulti, rPos, gPos, bPos, aPos;
bool alpha;

EXTERN_CVAR (Bool, viz_debug)
EXTERN_CVAR (Int, viz_screen_format)
EXTERN_CVAR (Bool, viz_depth)
EXTERN_CVAR (Bool, viz_labels)
EXTERN_CVAR (Bool, viz_automap)
EXTERN_CVAR (Bool, viz_nocheat)

bip::mapped_region *vizScreenSMRegion = NULL;
BYTE *vizScreen = NULL;

void VIZ_ScreenInit() {

    vizScreenSMRegion = NULL;
    VIZ_ScreenFormatUpdate();

    try {
        vizScreenSMRegion = new bip::mapped_region(vizSM, bip::read_write, vizSMScreenAddress, 10 * vizScreenChannelSize);
        vizScreen = static_cast<BYTE *>(vizScreenSMRegion->get_address());

        Printf("VIZ_ScreenInit: width: %d, height: %d, pitch: %zu, format: ",
               vizScreenWidth, vizScreenHeight, vizScreenPitch);

        switch(*viz_screen_format){
            case VIZ_SCREEN_CRCGCB:             Printf("CRCGCB\n"); break;
            case VIZ_SCREEN_RGB24:              Printf("RGB24\n"); break;
            case VIZ_SCREEN_RGBA32:             Printf("RGBA32\n"); break;
            case VIZ_SCREEN_ARGB32:             Printf("ARGB32\n"); break;
            case VIZ_SCREEN_CBCGCR:             Printf("CBCGCR\n"); break;
            case VIZ_SCREEN_BGR24:              Printf("BGR24\n"); break;
            case VIZ_SCREEN_BGRA32:             Printf("BGRA32\n"); break;
            case VIZ_SCREEN_ABGR32:             Printf("ABGR32\n"); break;
            case VIZ_SCREEN_GRAY8:              Printf("GRAY8\n"); break;
            case VIZ_SCREEN_DOOM_256_COLORS8:   Printf("DOOM_256_COLORS\n"); break;
            default:                            Printf("UNKNOWN\n");
        }

        if(*viz_screen_format > VIZ_SCREEN_DOOM_256_COLORS8)
            VIZ_ReportError("VIZ_ScreenInit", "Unknown screen format.");
    }
    catch(bip::interprocess_exception &ex){
        VIZ_ReportError("VIZ_ScreenInit", "Failed to create buffers.");
    }
}

void VIZ_ScreenFormatUpdate(){

    vizScreenWidth = (unsigned int) screen->GetWidth();
    vizScreenHeight = (unsigned int) screen->GetHeight();
    vizScreenChannelSize = sizeof(BYTE) * vizScreenWidth * vizScreenHeight;
    vizScreenSize = sizeof(BYTE) * vizScreenWidth * vizScreenHeight;
    vizScreenPitch = vizScreenWidth;

    switch(*viz_screen_format){
        case VIZ_SCREEN_CRCGCB :
            posMulti = 1;
            rPos = 0; gPos = (int)vizScreenSize; bPos = 2 * (int)vizScreenSize;
            alpha = false;
            vizScreenSize *= 3;
            break;

        case VIZ_SCREEN_RGB24 :
            vizScreenSize *= 3;
            vizScreenPitch *= 3;
            posMulti = 3;
            rPos = 2; gPos = 1; bPos = 0;
            alpha = false;
            break;

        case VIZ_SCREEN_RGBA32 :
            vizScreenSize *= 4;
            vizScreenPitch *= 4;
            posMulti = 4;
            rPos = 3, gPos = 2, bPos = 1;
            alpha = true; aPos = 0;
            break;

        case VIZ_SCREEN_ARGB32 :
            vizScreenSize *= 4;
            vizScreenPitch *= 4;
            posMulti = 4;
            rPos = 2, gPos = 1, bPos = 0;
            alpha = true; aPos = 3;
            break;

        case VIZ_SCREEN_CBCGCR :
            posMulti = 1;
            rPos = 2 * (int)vizScreenSize; gPos = (int)vizScreenSize, bPos = 0;
            alpha = false;
            vizScreenSize *= 3;
            break;

        case VIZ_SCREEN_BGR24 :
            vizScreenSize *= 3;
            vizScreenPitch *= 3;
            posMulti = 3;
            rPos = 0; gPos = 1; bPos = 2;
            alpha = false;
            break;

        case VIZ_SCREEN_BGRA32 :
            vizScreenSize *= 4;
            vizScreenPitch *= 4;
            posMulti = 4;
            rPos = 1; gPos = 2; bPos = 3;
            alpha = true; aPos = 0;
            break;

        case VIZ_SCREEN_ABGR32 :
            vizScreenSize *= 4;
            vizScreenPitch *= 4;
            posMulti = 4;
            rPos = 0; gPos = 1; bPos = 2;
            alpha = true; aPos = 3;
            break;

        default:
            break;
    }

    if(*viz_depth && !*viz_nocheat) {
        if(vizDepthMap!=NULL) delete vizDepthMap;
        vizDepthMap = new VIZDepthBuffer(vizScreenWidth, vizScreenHeight);
    }

    if(*viz_labels && !*viz_nocheat) {
        if(vizLabels!=NULL) delete vizLabels;
        vizLabels = new VIZLabelsBuffer(vizScreenWidth, vizScreenHeight);
    }
}

void VIZ_CopyBuffer(unsigned int startAddress){

    if(screen == NULL) return;
    screen->Lock(true);

    const BYTE *buffer = screen->GetBuffer();
    PalEntry *palette = screen->GetPalette();

    if(buffer == NULL || palette == NULL) return;

    const unsigned int screenSize = screen->GetWidth() * screen->GetHeight();
    const unsigned int bufferPitch = screen->GetPitch();
    const unsigned int screenWidth = screen->GetWidth();
    const unsigned int bufferPitchWidthDiff = bufferPitch - screenWidth;

    VIZ_DEBUG_PRINT("VIZ_CopyScreenBuffer: startAddress: %d, size: %d\n", startAddress, screenSize);

    if(vizScreenChannelSize != screenSize || vizScreenWidth != screenWidth)
        VIZ_ReportError("VIZ_CopyScreenBuffer", "Buffers size mismatch.");

    if(*viz_screen_format == VIZ_SCREEN_DOOM_256_COLORS8){
        for(unsigned int i = 0; i < screenSize; ++i){
            unsigned int b = i + (i / screenWidth) * bufferPitchWidthDiff;
            vizScreen[startAddress + i] = buffer[b];
        }
    }
    else if(*viz_screen_format == VIZ_SCREEN_GRAY8){
        for(unsigned int i = 0; i < screenSize; ++i){
            unsigned int b = i + (i / screenWidth) * bufferPitchWidthDiff;
            vizScreen[startAddress + i] = 0.21 * palette[buffer[b]].r + 0.72 * palette[buffer[b]].g + 0.07 * palette[buffer[b]].b;
        }
    }
    else {
        for (unsigned int i = 0; i < screenSize; ++i) {
            unsigned int pos = startAddress + i * posMulti;
            unsigned int b = i + (i / screenWidth) * bufferPitchWidthDiff;
            vizScreen[pos + rPos] = palette[buffer[b]].r;
            vizScreen[pos + gPos] = palette[buffer[b]].g;
            vizScreen[pos + bPos] = palette[buffer[b]].b;
            if (alpha) vizScreen[pos + aPos] = 255;
        }
    }

    screen->Unlock();
}

void VIZ_ScreenUpdate(){

    VIZ_CopyBuffer(SCREEN_BUFFER_SM_ADDRESS * vizScreenChannelSize);

    if(*viz_depth || *viz_labels) {

        screen->Lock(true);

        if (*viz_depth) {
            if (!*viz_nocheat && vizDepthMap!=NULL){
                memcpy(vizScreen + DEPTH_BUFFER_SM_ADDRESS * vizScreenChannelSize, vizDepthMap->getBuffer(), vizDepthMap->getBufferSize()); }
            else
                memset(vizScreen + DEPTH_BUFFER_SM_ADDRESS * vizScreenChannelSize, 0, DEPTH_BUFFER_SM_SIZE * vizScreenChannelSize);
        }

        if (*viz_labels) {
            if(!*viz_nocheat && vizLabels!=NULL) memcpy(vizScreen + LABELS_BUFFER_SM_ADDRESS * vizScreenChannelSize, vizLabels->getBuffer(), vizLabels->getBufferSize());
            else memset(vizScreen + LABELS_BUFFER_SM_ADDRESS * vizScreenChannelSize, 0, LABELS_BUFFER_SM_SIZE * vizScreenChannelSize);
        }

        screen->Unlock();
    }
}

void VIZ_ScreenLevelMapUpdate(){
    if(*viz_automap){
        if(!*viz_nocheat) VIZ_CopyBuffer(MAP_BUFFER_SM_ADDRESS * vizScreenChannelSize);
        else memset(vizScreen + MAP_BUFFER_SM_ADDRESS * vizScreenChannelSize, 0, MAP_BUFFER_SM_SIZE * vizScreenChannelSize);
    }
}

void VIZ_ScreenClose(){
    delete vizScreenSMRegion;
    if(vizDepthMap) delete vizDepthMap;
    if(vizLabels) delete vizLabels;
}