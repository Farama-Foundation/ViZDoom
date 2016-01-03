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

    DoomController *dc = new DoomController;

    std::cout << "\n\nVIZIA CONTROLLER EXAMPLE\n\n";

    dc->setGamePath("viziazdoom");
    dc->setIwadPath("../scenarios/doom2.wad");
    dc->setFilePath("../scenarios/s1_b.wad");
    dc->setMap("map01");
    dc->setMapTimeout(200);
    dc->setAutoMapRestart(true);
    dc->setSeed(131313);

    // w przypadku nie zachowania proporcji 4:3, 16:10, 16:9
    // silnik weźmie wysokość i pomnoży razy 4/3
    // możemy spróbować to wyłączyć, ale pewnie wtedy obraz będzie zniekształocny
    dc->setScreenResolution(640, 480);
    // rozdzielczość znacząco wpływa na szybkość działania

    dc->setScreenFormat(RGB24);

    dc->setRenderHud(true);
    dc->setRenderCrosshair(true);
    dc->setRenderWeapon(true);
    dc->setRenderDecals(true);
    dc->setRenderParticles(true);

    dc->setWindowHidden(true);
    dc->setNoXServer(false);

    dc->setNoConsole(false);

    dc->init();
    if(sdl) initSDL(dc->getScreenWidth(), dc->getScreenHeight());
    
    int loop = 100;
    for(int i = 0; i < 50000; ++i){

        if(dc->isMapLastTic()) std::cout << "\nMAP FINISHED\n\n";

        if(i%loop < 50) {
            dc->setButtonState(MOVE_RIGHT, 1);   //ustaw inpup
        }
        else{
            dc->setButtonState(MOVE_RIGHT, 0);
        }
        if(i%loop >= 50) {
            dc->getInput()->BT[MOVE_LEFT] = 1;  //lub w ten sposób
        }
        else{
            dc->getInput()->BT[MOVE_LEFT] = 0;
        }

        if(i%loop == 25 || i%loop == 50 || i%loop == 75){
            dc->setButtonState(ATTACK, 1);
        }
        else{
            dc->setButtonState(ATTACK, 0);
        }

        if(i%loop == 30 || i%loop == 60){
            dc->setButtonState(JUMP, 1);
        }
        else{
            dc->setButtonState(JUMP, 0);
        }

        if(i%10 == 0) {
            if (sdl)
                updateSDL(dc->getScreenWidth(), dc->getScreenHeight(), dc->getScreenPitch(),
                          (void *) dc->getScreen());

            std::cout << "GAME TIC: " << dc->getGameTic() << " MAP TIC: " << dc->getMapTic() <<
            " HP: " << dc->getPlayerHealth() << " ARMOR: " << dc->getGameVars()->PLAYER_ARMOR<<std::endl;
            std::cout << "ATTACK READY: " << dc->getGameVars()->PLAYER_ATTACK_READY << " SELECTED WEAPON: " << dc->getGameVars()->PLAYER_SELECTED_WEAPON << " SELECTED AMMO: " << dc->getGameVars()->PLAYER_SELECTED_WEAPON_AMMO << std::endl;
            for(int i = 0; i < 4; ++i){
                std::cout << "WEAPON " << i << ": " << dc->getGameVars()->PLAYER_WEAPON[i] << " AMMO " << i  << ": " << dc->getGameVars()->PLAYER_AMMO[i] << std::endl;
            }
            std::cout << "REWARD: " << dc->getMapReward() << " SHAPING: " << dc->getGameVar(USER1) << std::endl;
            dc->tic(true);
        }
        else dc->tic(false);

    }
    dc->close();
    if(sdl) closeSDL();
}

