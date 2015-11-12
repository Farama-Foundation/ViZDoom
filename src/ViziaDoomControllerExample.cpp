#include "ViziaDoomController.h"
#include <iostream>

int main(){

    ViziaDoomController *vdm = new ViziaDoomController;

    std::cout << "SETTING DOOM " << std::endl;

    vdm->setGamePath("./zdoom");
    vdm->setIwadPath("./dooom2.wad");
    vdm->setFilePath("./s1.wad");
    vdm->setMap("map01");

    std::cout << "STARTING DOOM " << std::endl;

    vdm->init();

    int loop = 100;
    for(int i = 0; i < 3000; ++i){

        vdm->setMouse(0, 0);

        if(i%loop < 50) {
            vdm->setButtonState(V_MOVERIGHT, true);   //ustaw inpup
        }
        else{
            vdm->setButtonState(V_MOVERIGHT, false);
        }
        if(i%loop >= 50) {
            vdm->getInput()->BT[V_MOVELEFT] = true;  //lub w ten sposób
        }
        else{
            vdm->getInput()->BT[V_MOVELEFT] = false;
        }

        if(i%loop == 25 || i%loop == 50 || i%loop == 75){
            vdm->setButtonState(V_ATTACK, true);
        }
        else{
            vdm->setButtonState(V_ATTACK, false);
        }

        if(i%loop == 30 || i%loop == 60){
            vdm->setButtonState(V_JUMP, true);
        }
        else{
            vdm->setButtonState(V_JUMP, false);
        }

        std::cout << "GAMETIC: " << vdm->getGameTic() << std::endl <<
                " HP: " << vdm->getPlayerHealth() << " AMMO: " << vdm->getGameVars()->PLAYER_AMMO[2] << std::endl;

        if(i == 1000 || i == 2000 || vdm->getPlayerHealth() <= 0){
            std::cout << "RESTART MAP " << std::endl;
            vdm->resetMap();
        }
        vdm->tic();
    }

    vdm->close();
}
