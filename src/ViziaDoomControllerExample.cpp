#include "ViziaDoomController.h"
#include <iostream>

int main(){

    ViziaDoomController *vdm = new ViziaDoomController;

    std::cout << "SETTING DOOM " << std::endl;

    vdm->setGamePath("zdoom");
    vdm->setIwadPath("doom2.wad");
    vdm->setFilePath("s1_b.wad");
    vdm->setMap("map01");
    vdm->setMapTimeout(300);

    // w przypadku nie zachowania proporcji 4:3, 16:10, 16:9
    // silnik weźmie wysokość i pomnoży razy 4/3
    // możemy spróbować to wyłączyć, ale pewnie wtedy obraz będzie zniekształocny
    vdm->setScreenSize(320, 240);
    // rozdzielczość znacząco wpływa na szybkość działania

    vdm->showHud(false);
    vdm->showCrosshair(true);
    vdm->showWeapon(true);
    vdm->showDecals(false);
    vdm->showParticles(false);

    vdm->init();
    int loop = 100;
    for(int i = 0; i < 1000; ++i){

        //vdm->setMouseX(-10); //obrót w lewo

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

        std::cout << "GAME TIC: " << vdm->getGameTic() << " MAP TIC: " << vdm->getMapTic() <<
                " HP: " << vdm->getPlayerHealth() << " AMMO: " << vdm->getGameVars()->PLAYER_AMMO[2] << std::endl;

        vdm->tic();
    }

    vdm->close();
}
