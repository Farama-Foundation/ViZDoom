#include "ViziaDoomController.h"

#include <iostream>

int main(){

    ViziaDoomController vdm;

    vdm.setGamePath("./zdoom");
    vdm.setIwadPath("./dooom2.wad");
    vdm.setFilePath("./s1.wad");
    vdm.setMap("MAP01");

    vdm.init();

    std::cout << " " BOOST_POSIX_API " " << std::endl;

    int timeout = 300; //10 sekund bo 35 ticków na sekunde :d
    for(int i = 0; i < 900; ++i){

        if((vdm.getGameTic()%timeout) < 150) {
            vdm.setButtonState(V_LEFT, true);   //ustaw input
            vdm.setButtonState(V_FORWARD, true);
        }
        else{
            vdm.setButtonState(V_LEFT, false);
            vdm.setButtonState(V_FORWARD, false);
        }
        if((vdm.getGameTic()%timeout) >= 150) {
            vdm.getInput()->BT[V_LEFT] = true;  //lub w ten sposób
            vdm.getInput()->BT[V_FORWARD] = true;
        }
        else{
            vdm.getInput()->BT[V_LEFT] = false;
            vdm.getInput()->BT[V_FORWARD] = false;
        }

        std::cout << vdm.getPlayerHealth() << " " << vdm.getGameVars()->PLAYER_AMMO[3] << std::endl;

        if(!vdm.getGameTic()%timeout || vdm.getPlayerHealth() <= 0) vdm.resetMap();
        vdm.tic();
    }

    vdm.close();
}
