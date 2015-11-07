#ifndef VIZIADOOMCONTROLER_H
#define VIZIADOOMCONTROLER_H

#include <string>
#include "SharedMemory.h"

class ViziaDoomControler {

    ViziaDoomControler();
    ViziaDoomControler(std::string game, std::string file, std::string map, int scr_w, int scr_h, int skill = 1);
    ~ViziaDoomControler();

    init();
    close();

    void setScreenSize(int scr_w, int scr_h);
    void setGame(std::string game);
    void setFile(std::string file);
    void setSkill(int skill);
    void setMap(std::string map);

    void changeMap(std::string map);
    void changeSkill(int skill);

    update();
    restart();

    uint8_t * getScreenPtr();
    ViziaInputSMStruct* getInputPtr();
    ViziaGameDataSMStruct* getGameDataPtr();

    int getGameTic();
    int getPlayerHealth();
    int getPlayerArmor();
    //..

    void setButton(int btn, bool set);
    void setBtAttack(bool set);
    void setBtLeft(bool set);
    void setBtRight(bool set);
    //..
};


#endif //VIZIADOOMCONTROLER_H
