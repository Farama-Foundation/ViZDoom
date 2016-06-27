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

#include "d_main.h"
#include "doomstat.h"
#include "v_video.h"

unsigned int vizScreenHeight;
unsigned int vizScreenWidth;
size_t vizScreenPitch;
size_t vizScreenSize;

int posMulti, rPos, gPos, bPos, aPos;
bool alpha;

EXTERN_CVAR (Bool, viz_debug)
EXTERN_CVAR (Int, viz_screen_format)
EXTERN_CVAR (Bool, viz_nocheat)

bip::mapped_region *vizScreenSMRegion = NULL;
BYTE *vizScreen = NULL;

void VIZ_ScreenInit() {

    vizScreenSMRegion = NULL;

    vizScreenWidth = (unsigned int) screen->GetWidth();
    vizScreenHeight = (unsigned int) screen->GetHeight();
    vizScreenSize = sizeof(BYTE) * vizScreenWidth * vizScreenHeight;
    vizScreenPitch = vizScreenWidth;

    switch(*viz_screen_format){
        case VIZ_SCREEN_CRCGCB :
            posMulti = 1;
            rPos = 0; gPos = (int)vizScreenSize; bPos = 2 * (int)vizScreenSize;
            alpha = false;
            vizScreenSize *= 3;
            break;

        case VIZ_SCREEN_CRCGCBDB :
            posMulti = 1;
            rPos = 0; gPos = (int)vizScreenSize; bPos = 2 * (int)vizScreenSize;
            alpha = false;
            vizScreenSize *= 4;
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

        case VIZ_SCREEN_CBCGCRDB :
            posMulti = 1;
            rPos = 2 * (int)vizScreenSize; gPos = (int)vizScreenSize, bPos = 0;
            alpha = false;
            vizScreenSize *= 4;
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

    try {
        vizScreenSMRegion = new bip::mapped_region(vizSM, bip::read_write, vizSMScreenAddress, vizScreenSize);
        vizScreen = static_cast<BYTE *>(vizScreenSMRegion->get_address());

        Printf("VIZ_ScreenInit: width: %d, height: %d, pitch: %zu, format: ",
               vizScreenWidth, vizScreenHeight, vizScreenPitch);

        switch(*viz_screen_format){
            case VIZ_SCREEN_CRCGCB:             Printf("CRCGCB\n"); break;
            case VIZ_SCREEN_CRCGCBDB:           Printf("CRCGCBDB\n"); break;
            case VIZ_SCREEN_RGB24:              Printf("RGB24\n"); break;
            case VIZ_SCREEN_RGBA32:             Printf("RGBA32\n"); break;
            case VIZ_SCREEN_ARGB32:             Printf("ARGB32\n"); break;
            case VIZ_SCREEN_CBCGCR:             Printf("CBCGCR\n"); break;
            case VIZ_SCREEN_CBCGCRDB:           Printf("CBCGCRDB\n"); break;
            case VIZ_SCREEN_BGR24:              Printf("BGR24\n"); break;
            case VIZ_SCREEN_BGRA32:             Printf("BGRA32\n"); break;
            case VIZ_SCREEN_ABGR32:             Printf("ABGR32\n"); break;
            case VIZ_SCREEN_GRAY8:              Printf("GRAY8\n"); break;
            case VIZ_SCREEN_DEPTH_BUFFER8:      Printf("DEPTH_BUFFER8\n"); break;
            case VIZ_SCREEN_DOOM_256_COLORS8:   Printf("DOOM_256_COLORS\n"); break;
            default:                            Printf("UNKNOWN\n");
        }
    }
    catch(bip::interprocess_exception &ex){
        Printf("VIZ_ScreenInit: Failed to create screen buffer.");
        VIZ_MQSend(VIZ_MSG_CODE_DOOM_ERROR, "Failed to create screen buffer.");
        exit(1);
    }

    if((*viz_screen_format==VIZ_SCREEN_CBCGCRDB
       ||*viz_screen_format==VIZ_SCREEN_CRCGCBDB
       ||*viz_screen_format==VIZ_SCREEN_DEPTH_BUFFER8) && !*viz_nocheat) {
        depthMap = new VIZDepthBuffer(vizScreenWidth, vizScreenHeight);
    }
}

void VIZ_ScreenUpdate(){

    screen->Lock(true);

    const BYTE *buffer = screen->GetBuffer();
    const unsigned int screenSize = screen->GetWidth() * screen->GetHeight();
    const unsigned int bufferPitch = screen->GetPitch();
    const unsigned int bufferWidth = screen->GetWidth();
    const unsigned int bufferPitchWidthDiff = bufferPitch - bufferWidth;

    if (buffer != NULL) {

        if(*viz_screen_format == VIZ_SCREEN_DOOM_256_COLORS8){
            for(unsigned int i = 0; i < screenSize; ++i){
                //
                unsigned int b = i + (i / bufferWidth) * bufferPitchWidthDiff;
                vizScreen[i] = buffer[b];
            }
        }
        else {
            PalEntry *palette;
            palette = screen->GetPalette();

            if(*viz_screen_format == VIZ_SCREEN_GRAY8){
                for(unsigned int i = 0; i < screenSize; ++i){
                    unsigned int b = i + (i / bufferWidth) * bufferPitchWidthDiff;
                    vizScreen[i] = 0.21 * palette[buffer[b]].r + 0.72 * palette[buffer[b]].g + 0.07 *palette[buffer[b]].b;
                }
            }
            else if(*viz_screen_format == VIZ_SCREEN_DEPTH_BUFFER8 && !*viz_nocheat){
                memcpy(vizScreen, depthMap->getBuffer(), depthMap->getBufferSize());
            }
            else {
                for (unsigned int i = 0; i < screenSize; ++i) {
                    unsigned int pos = i * posMulti;
                    unsigned int b = i + (i / bufferWidth) * bufferPitchWidthDiff;
                    vizScreen[pos + rPos] = palette[buffer[b]].r;
                    vizScreen[pos + gPos] = palette[buffer[b]].g;
                    vizScreen[pos + bPos] = palette[buffer[b]].b;
                    if (alpha) vizScreen[pos + aPos] = 255;
                }

                if((*viz_screen_format == VIZ_SCREEN_CRCGCBDB || *viz_screen_format == VIZ_SCREEN_CBCGCRDB) && !*viz_nocheat){
                    memcpy(vizScreen + 3*screenSize, depthMap->getBuffer(), depthMap->getBufferSize());
                }
            }
        }

    }
    screen->Unlock();
}

void VIZ_ScreenClose(){
    delete vizScreenSMRegion;
    delete depthMap;
}