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

#include "vizdoom_screen.h"
#include "vizdoom_defines.h"
#include "vizdoom_shared_memory.h"
#include "vizdoom_message_queue.h"
#include "vizdoom_depth.h"

#include "d_main.h"
#include "doomstat.h"
#include "v_video.h"

unsigned int vizdoomScreenHeight;
unsigned int vizdoomScreenWidth;
size_t vizdoomScreenPitch;
size_t vizdoomScreenSize;

int posMulti, rPos, gPos, bPos, aPos;
bool alpha;

EXTERN_CVAR (Int, vizdoom_screen_format)
EXTERN_CVAR (Bool, vizdoom_nocheat)

bip::mapped_region *vizdoomScreenSMRegion = NULL;
BYTE *vizdoomScreen = NULL;

void ViZDoom_ScreenInit() {

    vizdoomScreenSMRegion = NULL;

    vizdoomScreenWidth = (unsigned int) screen->GetWidth();
    vizdoomScreenHeight = (unsigned int) screen->GetHeight();
    vizdoomScreenSize = sizeof(BYTE) * vizdoomScreenWidth * vizdoomScreenHeight;
    //vizdoomScreenPitch = screen->GetPitch(); // returns 0 ??
    vizdoomScreenPitch = vizdoomScreenWidth;

    switch(*vizdoom_screen_format){
        case VIZDOOM_SCREEN_CRCGCB :
            posMulti = 1;
            rPos = 0; gPos = (int)vizdoomScreenSize; bPos = 2 * (int)vizdoomScreenSize;
            alpha = false;
            vizdoomScreenSize *= 3;
            break;

        case VIZDOOM_SCREEN_CRCGCBDB :
            posMulti = 1;
            rPos = 0; gPos = (int)vizdoomScreenSize; bPos = 2 * (int)vizdoomScreenSize;
            alpha = false;
            vizdoomScreenSize *= 4;
            break;

        case VIZDOOM_SCREEN_RGB24 :
            vizdoomScreenSize *= 3;
            vizdoomScreenPitch *= 3;
            posMulti = 3;
            rPos = 2; gPos = 1; bPos = 0;
            alpha = false;
            break;

        case VIZDOOM_SCREEN_RGBA32 :
            vizdoomScreenSize *= 4;
            vizdoomScreenPitch *= 4;
            posMulti = 4;
            rPos = 3, gPos = 2, bPos = 1;
            alpha = true; aPos = 0;
            break;

        case VIZDOOM_SCREEN_ARGB32 :
            vizdoomScreenSize *= 4;
            vizdoomScreenPitch *= 4;
            posMulti = 4;
            rPos = 2, gPos = 1, bPos = 0;
            alpha = true; aPos = 3;
            break;

        case VIZDOOM_SCREEN_CBCGCR :
            posMulti = 1;
            rPos = 2 * (int)vizdoomScreenSize; gPos = (int)vizdoomScreenSize, bPos = 0;
            alpha = false;
            vizdoomScreenSize *= 3;
            break;

        case VIZDOOM_SCREEN_CBCGCRDB :
            posMulti = 1;
            rPos = 2 * (int)vizdoomScreenSize; gPos = (int)vizdoomScreenSize, bPos = 0;
            alpha = false;
            vizdoomScreenSize *= 4;
            break;

        case VIZDOOM_SCREEN_BGR24 :
            vizdoomScreenSize *= 3;
            vizdoomScreenPitch *= 3;
            posMulti = 3;
            rPos = 0; gPos = 1; bPos = 2;
            alpha = false;
            break;

        case VIZDOOM_SCREEN_BGRA32 :
            vizdoomScreenSize *= 4;
            vizdoomScreenPitch *= 4;
            posMulti = 4;
            rPos = 1; gPos = 2; bPos = 3;
            alpha = true; aPos = 0;
            break;

        case VIZDOOM_SCREEN_ABGR32 :
            vizdoomScreenSize *= 4;
            vizdoomScreenPitch *= 4;
            posMulti = 4;
            rPos = 0; gPos = 1; bPos = 2;
            alpha = true; aPos = 3;
            break;

        default:
            break;
    }

    try {
        vizdoomScreenSMRegion = new bip::mapped_region(vizdoomSM, bip::read_write,
            sizeof(ViZDoomGameVarsStruct) + sizeof(ViZDoomInputStruct), vizdoomScreenSize);
        vizdoomScreen = static_cast<BYTE *>(vizdoomScreenSMRegion->get_address());

        Printf("ViZDoom_ScreenInit: width: %d, height: %d, pitch: %zu, format: ",
               vizdoomScreenWidth, vizdoomScreenHeight, vizdoomScreenPitch);

        switch(*vizdoom_screen_format){
            case VIZDOOM_SCREEN_CRCGCB: Printf("CRCGCB\n"); break;
            case VIZDOOM_SCREEN_CRCGCBDB: Printf("CRCGCBDB\n"); break;
            case VIZDOOM_SCREEN_RGB24: Printf("RGB24\n"); break;
            case VIZDOOM_SCREEN_RGBA32: Printf("RGBA32\n"); break;
            case VIZDOOM_SCREEN_ARGB32: Printf("ARGB32\n"); break;
            case VIZDOOM_SCREEN_CBCGCR: Printf("CBCGCR\n"); break;
            case VIZDOOM_SCREEN_CBCGCRDB: Printf("CBCGCRDB\n"); break;
            case VIZDOOM_SCREEN_BGR24: Printf("BGR24\n"); break;
            case VIZDOOM_SCREEN_BGRA32: Printf("BGRA32\n"); break;
            case VIZDOOM_SCREEN_ABGR32: Printf("ABGR32\n"); break;
            case VIZDOOM_SCREEN_GRAY8: Printf("GRAY8\n"); break;
            case VIZDOOM_SCREEN_DEPTH_BUFFER8: Printf("DEPTH_BUFFER8\n"); break;
            case VIZDOOM_SCREEN_DOOM_256_COLORS8: Printf("DOOM_256_COLORS\n"); break;
            default: Printf("UNKNOWN\n");
        }
    }
    catch(bip::interprocess_exception &ex){
        Printf("ViZDoom_ScreenInit: Error creating ViziaScreen SM");
        ViZDoom_MQSend(VIZDOOM_MSG_CODE_DOOM_ERROR);
        exit(1);
    }

    if((*vizdoom_screen_format==VIZDOOM_SCREEN_CBCGCRDB
       ||*vizdoom_screen_format==VIZDOOM_SCREEN_CRCGCBDB
       ||*vizdoom_screen_format==VIZDOOM_SCREEN_DEPTH_BUFFER8) && !*vizdoom_nocheat) {
        depthMap = new ViZDoomDepthBuffer(vizdoomScreenWidth, vizdoomScreenHeight);
    }
}

void ViZDoom_ScreenUpdate(){

    screen->Lock(true);

    const BYTE *buffer = screen->GetBuffer();
    const unsigned int bufferSize = screen->GetWidth() * screen->GetHeight();

    if (buffer != NULL) {

        if(*vizdoom_screen_format == VIZDOOM_SCREEN_DOOM_256_COLORS8){
            for(unsigned int i = 0; i < bufferSize; ++i){
                vizdoomScreen[i] = buffer[i];
            }
        }
        else {
            PalEntry *palette;
            palette = screen->GetPalette();

            if(*vizdoom_screen_format == VIZDOOM_SCREEN_GRAY8){
                for(unsigned int i = 0; i < bufferSize; ++i){
                    vizdoomScreen[i] = 0.21 * palette[buffer[i]].r + 0.72 * palette[buffer[i]].g + 0.07 *palette[buffer[i]].b;
                }
            }
            else if(*vizdoom_screen_format == VIZDOOM_SCREEN_DEPTH_BUFFER8 && !*vizdoom_nocheat){
                memcpy(vizdoomScreen, depthMap->getBuffer(), depthMap->getBufferSize());
            }
            else {
                for (unsigned int i = 0; i < bufferSize; ++i) {
                    unsigned int pos = i * posMulti;
                    vizdoomScreen[pos + rPos] = palette[buffer[i]].r;
                    vizdoomScreen[pos + gPos] = palette[buffer[i]].g;
                    vizdoomScreen[pos + bPos] = palette[buffer[i]].b;
                    if (alpha) vizdoomScreen[pos + aPos] = 255;
                    //if(alpha) vizdoomScreen[pos + aPos] = palette[buffer[i]].a;
                }

                if((*vizdoom_screen_format == VIZDOOM_SCREEN_CRCGCBDB || *vizdoom_screen_format == VIZDOOM_SCREEN_CBCGCRDB) && !*vizdoom_nocheat){
                    memcpy(vizdoomScreen+3*bufferSize, depthMap->getBuffer(), depthMap->getBufferSize());
                }
            }
        }

        //memcpy( vizdoomScreen, screen->GetBuffer(), vizdoomScreenSize );
    }
    screen->Unlock();
}

void ViZDoom_ScreenClose(){
    delete vizdoomScreenSMRegion;
    delete depthMap;
}