#include "vizia_screen.h"
#include "vizia_defines.h"
#include "vizia_shared_memory.h"
#include "vizia_message_queue.h"
#include "vizia_depth.h"

#include "d_main.h"
#include "d_net.h"
#include "g_game.h"
#include "doomdef.h"
#include "doomstat.h"

#include "v_video.h"
#include "r_renderer.h"

unsigned int viziaScreenHeight;
unsigned int viziaScreenWidth;
size_t viziaScreenPitch;
size_t viziaScreenSize;

int posMulti, rPos, gPos, bPos, aPos;
bool alpha;

EXTERN_CVAR (Int, vizia_screen_format)

bip::mapped_region *viziaScreenSMRegion = NULL;
BYTE *viziaScreen = NULL;

void Vizia_ScreenInit() {

    viziaScreenSMRegion = NULL;

    viziaScreenWidth = (unsigned int) screen->GetWidth();
    viziaScreenHeight = (unsigned int) screen->GetHeight();
    viziaScreenSize = sizeof(BYTE) * viziaScreenWidth * viziaScreenHeight;
    //viziaScreenPitch = screen->GetPitch(); // returns 0 ??
    viziaScreenPitch = viziaScreenWidth;

    switch(*vizia_screen_format){
        case VIZIA_SCREEN_CRCGCB :
            posMulti = 1;
            rPos = 0; gPos = (int)viziaScreenSize; bPos = 2 * (int)viziaScreenSize;
            alpha = false;
            viziaScreenSize *= 3;
            break;

        case VIZIA_SCREEN_CRCGCBDB :
            posMulti = 1;
            rPos = 0; gPos = (int)viziaScreenSize; bPos = 2 * (int)viziaScreenSize;
            alpha = false;
            viziaScreenSize *= 4;
            break;

        case VIZIA_SCREEN_RGB24 :
            viziaScreenSize *= 3;
            viziaScreenPitch *= 3;
            posMulti = 3;
            rPos = 2; gPos = 1; bPos = 0;
            alpha = false;
            break;

        case VIZIA_SCREEN_RGBA32 :
            viziaScreenSize *= 4;
            viziaScreenPitch *= 4;
            posMulti = 4;
            rPos = 3, gPos = 2, bPos = 1;
            alpha = true; aPos = 0;
            break;

        case VIZIA_SCREEN_ARGB32 :
            viziaScreenSize *= 4;
            viziaScreenPitch *= 4;
            posMulti = 4;
            rPos = 2, gPos = 1, bPos = 0;
            alpha = true; aPos = 3;
            break;

        case VIZIA_SCREEN_CBCGCR :
            posMulti = 1;
            rPos = 2 * (int)viziaScreenSize; gPos = (int)viziaScreenSize, bPos = 0;
            alpha = false;
            viziaScreenSize *= 3;
            break;

        case VIZIA_SCREEN_CBCGCRDB :
            posMulti = 1;
            rPos = 2 * (int)viziaScreenSize; gPos = (int)viziaScreenSize, bPos = 0;
            alpha = false;
            viziaScreenSize *= 4;
            break;

        case VIZIA_SCREEN_BGR24 :
            viziaScreenSize *= 3;
            viziaScreenPitch *= 3;
            posMulti = 3;
            rPos = 0; gPos = 1; bPos = 2;
            alpha = false;
            break;

        case VIZIA_SCREEN_BGRA32 :
            viziaScreenSize *= 4;
            viziaScreenPitch *= 4;
            posMulti = 4;
            rPos = 1; gPos = 2; bPos = 3;
            alpha = true; aPos = 0;
            break;

        case VIZIA_SCREEN_ABGR32 :
            viziaScreenSize *= 4;
            viziaScreenPitch *= 4;
            posMulti = 4;
            rPos = 0; gPos = 1; bPos = 2;
            alpha = true; aPos = 3;
            break;

        default:
            break;
    }

    try {
        viziaScreenSMRegion = new bip::mapped_region(viziaSM, bip::read_write,
            sizeof(ViziaInputStruct) + sizeof(ViziaGameVarsStruct), viziaScreenSize);
        viziaScreen = static_cast<BYTE *>(viziaScreenSMRegion->get_address());

        Printf("Vizia_ScreenInit: width: %d, height: %d, pitch: %zu, format: ",
               viziaScreenWidth, viziaScreenHeight, viziaScreenPitch);

        switch(*vizia_screen_format){
            case VIZIA_SCREEN_CRCGCB: Printf("CRCGCB\n"); break;
            case VIZIA_SCREEN_CRCGCBDB: Printf("CRCGCBDB\n"); break;
            case VIZIA_SCREEN_RGB24: Printf("RGB24\n"); break;
            case VIZIA_SCREEN_RGBA32: Printf("RGBA32\n"); break;
            case VIZIA_SCREEN_ARGB32: Printf("ARGB32\n"); break;
            case VIZIA_SCREEN_CBCGCR: Printf("CBCGCR\n"); break;
            case VIZIA_SCREEN_CBCGCRDB: Printf("CBCGCRDB\n"); break;
            case VIZIA_SCREEN_BGR24: Printf("BGR24\n"); break;
            case VIZIA_SCREEN_BGRA32: Printf("BGRA32\n"); break;
            case VIZIA_SCREEN_ABGR32: Printf("ABGR32\n"); break;
            case VIZIA_SCREEN_GRAY8: Printf("GRAY8\n"); break;
            case VIZIA_SCREEN_DEPTH_BUFFER8: Printf("DEPTH_BUFFER8\n"); break;
            case VIZIA_SCREEN_DOOM_256_COLORS8: Printf("DOOM_256_COLORS\n"); break;
            default: Printf("UNKNOWN\n");
        }
    }
    catch(bip::interprocess_exception &ex){
        Printf("Vizia_ScreenInit: Error creating ViziaScreen SM");
        Vizia_MQSend(VIZIA_MSG_CODE_DOOM_ERROR);
        exit(1);
    }

    if(*vizia_screen_format==VIZIA_SCREEN_CBCGCRDB
       ||*vizia_screen_format==VIZIA_SCREEN_CRCGCBDB
       ||*vizia_screen_format==VIZIA_SCREEN_DEPTH_BUFFER8) {
        depthMap = new depthBuffer(viziaScreenWidth, viziaScreenHeight);
        depthMap->setDepthBoundries(120000000, 358000);//SHOULDN'T BE HERE!
    }
}

void Vizia_ScreenUpdate(){

    screen->Lock(true);

    const BYTE *buffer = screen->GetBuffer();
    const int bufferSize = screen->GetWidth() * screen->GetHeight();

    if (buffer != NULL) {

        if(*vizia_screen_format == VIZIA_SCREEN_DOOM_256_COLORS8){
            for(unsigned int i = 0; i < bufferSize; ++i){
                viziaScreen[i] = buffer[i];
            }
        }
        else {
            PalEntry *palette;
            palette = screen->GetPalette();

            if(*vizia_screen_format == VIZIA_SCREEN_GRAY8){
                for(unsigned int i = 0; i < bufferSize; ++i){
                    viziaScreen[i] = 0.21 * palette[buffer[i]].r + 0.72 * palette[buffer[i]].g + 0.07 *palette[buffer[i]].b;
                }
            }
            else if(*vizia_screen_format == VIZIA_SCREEN_DEPTH_BUFFER8){
                memcpy(viziaScreen, depthMap->getBuffer(), depthMap->getBufferSize());
            }
            else {
                for (unsigned int i = 0; i < bufferSize; ++i) {
                    unsigned int pos = i * posMulti;
                    viziaScreen[pos + rPos] = palette[buffer[i]].r;
                    viziaScreen[pos + gPos] = palette[buffer[i]].g;
                    viziaScreen[pos + bPos] = palette[buffer[i]].b;
                    if (alpha) viziaScreen[pos + aPos] = 255;
                    //if(alpha) viziaScreen[pos + aPos] = palette[buffer[i]].a;
                }

                if(*vizia_screen_format == VIZIA_SCREEN_CRCGCBDB || *vizia_screen_format == VIZIA_SCREEN_CBCGCRDB){
                    memcpy(viziaScreen+3*bufferSize, depthMap->getBuffer(), depthMap->getBufferSize());
                }
            }
        }

        //memcpy( viziaScreen, screen->GetBuffer(), viziaScreenSize );
    }
    screen->Unlock();
}

void Vizia_ScreenClose(){
    delete(viziaScreenSMRegion);
    delete depthMap;
}