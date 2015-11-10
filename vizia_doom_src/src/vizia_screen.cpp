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

bip::mapped_region *viziaScreenSMRegion;
BYTE *viziaScreen;

void Vizia_ScreenInit() {
    viziaScreenWidth = screen->GetWidth();
    viziaScreenHeight = screen->GetHeight();
    viziaScreenPitch = screen->GetPitch();
    viziaScreenSize = sizeof(BYTE) * viziaScreenWidth * viziaScreenHeight;

    viziaScreenSMRegion = new bip::mapped_region(viziaSM, bip::read_write, Vizia_SMGetScreenRegionBeginning(), viziaScreenSize);
    viziaScreen = static_cast<BYTE *>(viziaScreenSMRegion->get_address());

    viziaScreen[0] = 2;
    viziaScreen[viziaScreenSize+100] = 4;

    printf("Screen SM region size: %zu, beginnig: %p, end: %p \n",
           viziaScreenSMRegion->get_size(), viziaScreenSMRegion->get_address(), viziaScreenSMRegion->get_address() + viziaScreenSMRegion->get_size());
}

void Vizia_ScreenUpdate(){

    if (gamestate == GS_LEVEL && !paused) {

        if (screen->GetWidth() != viziaScreenWidth || screen->GetHeight() != viziaScreenHeight) {
            viziaScreenWidth = screen->GetWidth();
            viziaScreenHeight = screen->GetHeight();
            viziaScreenPitch = screen->GetPitch();

            viziaScreenSize = sizeof(BYTE) * viziaScreenWidth * viziaScreenHeight;
            Vizia_SMSetSize(viziaScreenWidth, viziaScreenHeight);
        }

        screen->Lock(true);
        //const BYTE *buffer = screen->GetBuffer();
        if (screen->GetBuffer() != NULL) {
            memcpy( viziaScreen, screen->GetBuffer(), viziaScreenSize );
        }
        screen->Unlock();
    }
}

void Vizia_ScreenClose(){
    delete(viziaScreenSMRegion);
}