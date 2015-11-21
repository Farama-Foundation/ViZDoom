#include "vizia_screen.h"
#include "vizia_shared_memory.h"

#include "d_main.h"
#include "d_net.h"
#include "g_game.h"
#include "doomdef.h"
#include "doomstat.h"

#include "v_video.h"
#include "r_renderer.h"

int viziaScreenHeight;
int viziaScreenWidth;
int viziaScreenPitch;
size_t viziaScreenSize;

int posMulti, rPos, gPos, bPos, aPos;
bool alpha;


EXTERN_CVAR (Int, vizia_screen_format)

bip::mapped_region *viziaScreenSMRegion;
BYTE *viziaScreen;

void Vizia_ScreenInit() {
    viziaScreenWidth = screen->GetWidth();
    viziaScreenHeight = screen->GetHeight();
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

        case VIZIA_SCREEN_CRCGCBCA :
            posMulti = 1;
            rPos = 0; gPos = (int)viziaScreenSize; bPos = 2 * (int)viziaScreenSize;
            alpha = true; aPos = 3 * (int)viziaScreenSize;
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

        case VIZIA_SCREEN_CBCGCRCA :
            posMulti = 1;
            rPos = 2 * (int)viziaScreenSize; gPos = (int)viziaScreenSize, bPos = 0;
            alpha = true; aPos = 3 * (int)viziaScreenSize;
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

    viziaScreenSMRegion = new bip::mapped_region(viziaSM, bip::read_write, Vizia_SMGetScreenRegionBeginning(), viziaScreenSize);
    viziaScreen = static_cast<BYTE *>(viziaScreenSMRegion->get_address());

    printf("Vizia_ScreenInit: ViziaScreen SM region size: %zu, beginnig: %p, end: %p \n",
           viziaScreenSMRegion->get_size(), viziaScreenSMRegion->get_address(), viziaScreenSMRegion->get_address() + viziaScreenSMRegion->get_size());
    printf("Vizia_ScreenInit: width: %d, height: %d, pitch: %d, format, %d\n",
           viziaScreenWidth, viziaScreenHeight, viziaScreenPitch, *vizia_screen_format);

}

void Vizia_ScreenUpdate(){

    if (gamestate == GS_LEVEL && !paused) {

        screen->Lock(true);

        const BYTE *buffer = screen->GetBuffer();
        const int bufferSize = screen->GetWidth() * screen->GetHeight();

        if (buffer != NULL) {

            PalEntry *palette;
            palette = screen->GetPalette ();

            for(uint i = 0; i < bufferSize; ++i){
                uint pos = i * posMulti;
                viziaScreen[pos + rPos] = palette[buffer[i]].r;
                viziaScreen[pos + gPos] = palette[buffer[i]].g;
                viziaScreen[pos + bPos] = palette[buffer[i]].b;
                if(alpha) viziaScreen[pos + aPos] = 255;
                //if(alpha) viziaScreen[pos + aPos] = palette[buffer[i]].a;
            }

            //memcpy( viziaScreen, screen->GetBuffer(), viziaScreenSize );
        }
        screen->Unlock();
    }
}

void Vizia_ScreenClose(){
    delete(viziaScreenSMRegion);
}