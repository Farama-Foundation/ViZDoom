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
//shared_memory_object *viziaScreenSM;
BYTE *viziaScreen;

void Vizia_ScreenInit() {
    viziaScreenWidth = screen->GetWidth();
    viziaScreenHeight = screen->GetHeight();
    viziaScreenPitch = screen->GetPitch();
    viziaScreenSize = sizeof(BYTE) * viziaScreenWidth * viziaScreenHeight;

    //viziaScreenSM = new shared_memory_object(open_or_create, VIZIA_SCREEN_SM_NAME, read_write);
    //viziaScreenSM->truncate(viziaScreenSize);
    //mapped_region viziaScreenSMRegion(viziaScreenSM, read_write);

    mapped_region viziaScreenSMRegion(viziaSM, read_write, Vizia_SMGetScreenRegionBeginning(), viziaScreenSize);
    viziaScreen = static_cast<BYTE *>(viziaScreenSMRegion.get_address());
    memset(viziaScreen, 0, viziaScreenSize);
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
        const BYTE *buffer = screen->GetBuffer();
        if (buffer != NULL) {
            memcpy( viziaScreen, buffer, viziaScreenSize );
        }
        screen->Unlock();
    }
}

void Vizia_ScreenClose(){
    //shared_memory_object::remove(VIZIA_SCREEN_SM_NAME);
}