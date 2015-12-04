#include "ViziaDoomController.h"
#include <iostream>
#include <SDL2/SDL.h>

using namespace Vizia;

SDL_Window* window = NULL;
SDL_Surface* screen = NULL;
SDL_Surface* viziaBuffer = NULL;

void initSDL(int scrW, int scrH){
    SDL_Init( SDL_INIT_VIDEO );
    window = SDL_CreateWindow( "Vizia Example", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, scrW, scrH, SDL_WINDOW_SHOWN );
    screen = SDL_GetWindowSurface( window );
}

void updateSDL(int scrW, int scrH, int pitch, void* bufferPointer){
    viziaBuffer = SDL_CreateRGBSurfaceFrom(bufferPointer, scrW, scrH, 24, pitch, 0, 0, 0, 0);
    SDL_BlitSurface( viziaBuffer, NULL, screen, NULL );
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
    vdm->setFilePath("s1_b.wad");
    vdm->setMap("map01");
    vdm->setMapTimeout(200);
    vdm->setAutoMapRestart(true);

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

    vdm->init();
    if(sdl) initSDL(vdm->getScreenWidth(), vdm->getScreenHeight());
    
    int loop = 100;
    for(int i = 0; i < 50000; ++i){

        //vdm->setMouseX(-10); //obrót w lewo

        if(i%loop < 50) {
            vdm->setButtonState(MOVERIGHT, true);   //ustaw inpup
        }
        else{
            vdm->setButtonState(MOVERIGHT, false);
        }
        if(i%loop >= 50) {
            vdm->getInput()->BT[MOVELEFT] = true;  //lub w ten sposób
        }
        else{
            vdm->getInput()->BT[MOVELEFT] = false;
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

        std::cout << "GAME TIC: " << vdm->getGameTic() << " MAP TIC: " << vdm->getMapTic() <<
                " HP: " << vdm->getPlayerHealth() << " AMMO: " << vdm->getGameVars()->PLAYER_AMMO[2] <<
                " REWARD: " << vdm->getMapReward() << " SHAPING: " << vdm->getMapShapingReward() << std::endl;

        if(vdm->isMapLastTic()) std::cout << "\nMAP FINISHED\n\n";

        if(sdl) updateSDL(vdm->getScreenWidth(), vdm->getScreenHeight(), vdm->getScreenPitch(), (void*)vdm->getScreen());

        vdm->tic();
    }

    vdm->close();
    if(sdl) closeSDL();
}

