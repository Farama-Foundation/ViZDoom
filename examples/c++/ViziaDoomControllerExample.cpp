#include "ViziaDoomController.h"
#include <iostream>
#include <SDL2/SDL.h>

using namespace Vizia;

SDL_Window* window = NULL;
SDL_Surface* screen = NULL;
SDL_Surface* viziaBuffer = NULL;
SDL_Surface* viziaDepth = NULL;
unsigned char dBf[320][240];

void initSDL(int scrW, int scrH){
    SDL_Init( SDL_INIT_VIDEO );
    window = SDL_CreateWindow( "Vizia Example", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, 2*scrW, scrH, SDL_WINDOW_SHOWN );
    screen = SDL_GetWindowSurface( window );
    //BEGIN TEMP
    for(int i = 0;i< 320; i++)
        for(int j=0;j<240; j++)
            dBf[i][j]= j;
    //END TEMP
}

void updateSDL(int scrW, int scrH, int pitch, void* bufferPointer){
    SDL_Color colors[256];
    SDL_Rect ptrs;
    int i;

    for(i = 0; i < 256; i++)
    {
        colors[i].r = colors[i].g = colors[i].b = i;
    }
    viziaDepth = SDL_CreateRGBSurfaceFrom(dBf, scrW, scrH, 8, scrW, 0, 0, 0, 0);

    SDL_SetPaletteColors(viziaDepth->format->palette, colors, 0, 256);

    viziaBuffer = SDL_CreateRGBSurfaceFrom(bufferPointer, scrW, scrH, 24, pitch, 0, 0, 0, 0);
    printf("A\n");
    ptrs.x=0;
    ptrs.y=0;

    SDL_BlitSurface( viziaBuffer, NULL, screen, &ptrs );
    printf("C\n");
    ptrs.x = scrW;
    SDL_BlitSurface( viziaDepth, NULL, screen, &ptrs);
    printf("b\n");
    SDL_UpdateWindowSurface( window );
}

void closeSDL(){
    SDL_DestroyWindow( window );
    window = NULL;
    SDL_FreeSurface( screen);
    screen = NULL;
    SDL_FreeSurface( viziaBuffer );
    viziaBuffer = NULL;
    SDL_Quit();
}

int main(){

    bool sdl = true;

    DoomController *vdm = new DoomController;

    std::cout << "\n\nVIZIA CONTROLLER EXAMPLE\n\n";

    vdm->setGamePath("viziazdoom");
    vdm->setIwadPath("doom2.wad");
    vdm->setFilePath("../scenarios/s1_b.wad");
    vdm->setMap("map01");
    vdm->setMapTimeout(200);
    vdm->setAutoMapRestart(true);
    vdm->setSeed(131313);

    // w przypadku nie zachowania proporcji 4:3, 16:10, 16:9
    // silnik weźmie wysokość i pomnoży razy 4/3
    // możemy spróbować to wyłączyć, ale pewnie wtedy obraz będzie zniekształocny
    vdm->setScreenResolution(320, 240);
    // rozdzielczość znacząco wpływa na szybkość działania

    vdm->setScreenFormat(RGB24);

    vdm->setRenderHud(true);
    vdm->setRenderCrosshair(true);
    vdm->setRenderWeapon(true);
    vdm->setRenderDecals(true);
    vdm->setRenderParticles(true);

    vdm->setWindowHidden(false);
    vdm->setNoXServer(true);

    vdm->setNoConsole(true);

    vdm->init();
    if(sdl) initSDL(vdm->getScreenWidth(), vdm->getScreenHeight());
    
    int loop = 100;
    for(int i = 0; i < 50000; ++i){

        if(vdm->isMapLastTic()) std::cout << "\nMAP FINISHED\n\n";

        //vdm->setMouseX(-10); //obrót w lewo

        if(i%loop < 50) {
            vdm->setButtonState(MOVE_RIGHT, true);   //ustaw inpup
        }
        else{
            vdm->setButtonState(MOVE_RIGHT, false);
        }
        if(i%loop >= 50) {
            vdm->getInput()->BT[MOVE_LEFT] = true;  //lub w ten sposób
        }
        else{
            vdm->getInput()->BT[MOVE_LEFT] = false;
        }

        if(i%loop == 25 || i%loop == 50 || i%loop == 75){
            vdm->setButtonState(ATTACK, true);
        }
        else{
            vdm->setButtonState(ATTACK, false);
        }

        if(i%loop == 30 || i%loop == 60){
            vdm->setButtonState(JUMP, true);
        }
        else{
            vdm->setButtonState(JUMP, false);
        }

        if(i%10 == 0) {
            vdm->tic(true);
            if (sdl)
                updateSDL(vdm->getScreenWidth(), vdm->getScreenHeight(), vdm->getScreenPitch(),
                          (void *) vdm->getScreen());

            std::cout << "GAME TIC: " << vdm->getGameTic() << " MAP TIC: " << vdm->getMapTic() <<
            " HP: " << vdm->getPlayerHealth() << " AMMO: " << vdm->getGameVars()->PLAYER_AMMO[0] <<
            " REWARD: " << vdm->getMapReward() << " SHAPING: " << vdm->getGameVar(USER1) << std::endl;
        }
        else vdm->tic(false);

    }

    vdm->close();
    if(sdl) closeSDL();
}

