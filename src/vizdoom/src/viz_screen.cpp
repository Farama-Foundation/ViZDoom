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
#include "viz_depth.h"
#include "viz_labels.h"
#include "viz_main.h"

unsigned int vizScreenWidth, vizScreenHeight;
size_t vizScreenPitch, vizScreenSize, vizScreenChannelSize;

int posMulti, rPos, gPos, bPos, aPos;
bool alpha;

BYTE *vizScreenSM = NULL, *vizDepthSM = NULL, *vizLabelsSM = NULL, *vizAutomapSM = NULL;

EXTERN_CVAR (Bool, viz_debug)
EXTERN_CVAR (Int, viz_screen_format)
EXTERN_CVAR (Bool, viz_depth)
EXTERN_CVAR (Bool, viz_labels)
EXTERN_CVAR (Bool, viz_automap)
EXTERN_CVAR (Bool, viz_nocheat)

void VIZ_ScreenInit() {

    VIZ_ScreenFormatUpdate();
    VIZ_ScreenUpdateSM();

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
}

void VIZ_ScreenFormatUpdate(){

    vizScreenWidth = (unsigned int) screen->GetWidth();
    vizScreenHeight = (unsigned int) screen->GetHeight();
    vizScreenChannelSize = sizeof(BYTE) * vizScreenWidth * vizScreenHeight;
    vizScreenSize = vizScreenChannelSize;
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

        case VIZ_SCREEN_GRAY8:
            break;

        case VIZ_SCREEN_DOOM_256_COLORS8:
            break;

        default:
            VIZ_Error(VIZ_FUNC, "Unknown screen format.");
    }

    if (vizDepthMap != NULL){
        delete vizDepthMap;
        vizDepthMap = NULL;
    }
    if (vizLabels != NULL){
        delete vizLabels;
        vizLabels = NULL;
    }

    if(!*viz_nocheat) {
        if (*viz_depth) vizDepthMap = new VIZDepthBuffer(vizScreenWidth, vizScreenHeight);
        if (*viz_labels) vizLabels = new VIZLabelsBuffer(vizScreenWidth, vizScreenHeight);
    }
}

void VIZ_ScreenUpdateSM(){

    size_t SMBufferSize[4] = {vizScreenSize, 0, 0, 0};
    size_t SMBuffersSize = vizScreenSize;
    if (*viz_depth){
        SMBuffersSize += vizScreenChannelSize;
        SMBufferSize[1] = vizScreenChannelSize;
    }
    if (*viz_labels){
        SMBuffersSize += vizScreenChannelSize;
        SMBufferSize[2] = vizScreenChannelSize;
    }
    if (*viz_automap){
        SMBuffersSize += vizScreenSize;
        SMBufferSize[3] = vizScreenSize;
    }

    VIZ_SMUpdate(SMBuffersSize);

    try {
        for (int i = 0; i != 4; ++i) {
            VIZSMRegion *bufferRegion = &vizSMRegion[i + 2];
            if (SMBufferSize[i]) {
                VIZ_SMCreateRegion(bufferRegion, false, VIZ_SMGetRegionOffset(bufferRegion), SMBufferSize[i]);
                memset(bufferRegion->address, 0, bufferRegion->size);
            }
            else VIZ_SMDeleteRegion(bufferRegion);

            VIZ_DebugMsg(1, VIZ_FUNC, "region: %d, offset %zu, size: %zu", i + 2,
                         bufferRegion->offset, bufferRegion->size);
        }
    }
    catch(...){ // bip::interprocess_exception
        VIZ_Error(VIZ_FUNC, "Failed to map buffers.");
    }

    vizScreenSM = static_cast<BYTE *>(vizSMRegion[2].address);
    vizDepthSM = static_cast<BYTE *>(vizSMRegion[3].address);
    vizLabelsSM = static_cast<BYTE *>(vizSMRegion[4].address);
    vizAutomapSM = static_cast<BYTE *>(vizSMRegion[5].address);
}

void VIZ_CopyBuffer(BYTE *vizBuffer){

    if(screen == NULL) return;

    const BYTE *buffer = screen->GetBuffer();
    PalEntry *palette = screen->GetPalette();

    if(buffer == NULL || palette == NULL) return;

    const unsigned int screenSize = screen->GetWidth() * screen->GetHeight();
    const unsigned int bufferPitch = screen->GetPitch();
    const unsigned int screenWidth = screen->GetWidth();
    const unsigned int bufferPitchWidthDiff = bufferPitch - screenWidth;

    VIZ_DebugMsg(3, VIZ_FUNC, "bufferSize: %d, screenSize: %d", vizScreenSize, screenSize);

    if(vizScreenChannelSize != screenSize || vizScreenWidth != screenWidth)
        VIZ_Error(VIZ_FUNC, "Buffers size mismatch.");

    if(*viz_screen_format == VIZ_SCREEN_DOOM_256_COLORS8){
        for(unsigned int i = 0; i < screenSize; ++i){
            unsigned int b = i + (i / screenWidth) * bufferPitchWidthDiff;
            vizBuffer[i] = buffer[b];
        }
    }
    else if(*viz_screen_format == VIZ_SCREEN_GRAY8){
        for(unsigned int i = 0; i < screenSize; ++i){
            unsigned int b = i + (i / screenWidth) * bufferPitchWidthDiff;
            vizBuffer[i] = 0.21 * palette[buffer[b]].r + 0.72 * palette[buffer[b]].g + 0.07 * palette[buffer[b]].b;
        }
    }
    else {
        for (unsigned int i = 0; i < screenSize; ++i) {
            unsigned int pos = i * posMulti;
            unsigned int b = i + (i / screenWidth) * bufferPitchWidthDiff;
            vizBuffer[pos + rPos] = palette[buffer[b]].r;
            vizBuffer[pos + gPos] = palette[buffer[b]].g;
            vizBuffer[pos + bPos] = palette[buffer[b]].b;
            if (alpha) vizBuffer[pos + aPos] = 255;
        }
    }


}

void VIZ_ScreenUpdate(){
    screen->Lock(true);

    VIZ_CopyBuffer(vizScreenSM);

    if (*viz_depth && vizDepthMap != NULL)
        memcpy(vizDepthSM, vizDepthMap->getBuffer(), vizDepthMap->getBufferSize());

    if (*viz_labels && vizLabels != NULL)
        memcpy(vizLabelsSM, vizLabels->getBuffer(), vizLabels->getBufferSize());

    screen->Unlock();
}

void VIZ_ScreenLevelMapUpdate(){
    screen->Lock(true);
    if(*viz_automap) VIZ_CopyBuffer(vizAutomapSM);
    screen->Unlock();
}

void VIZ_ScreenClose(){
    if(vizDepthMap) delete vizDepthMap;
    if(vizLabels) delete vizLabels;
}