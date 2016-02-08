#include "ViziaDoomGame.h"
#include <iostream>
#include <vector>

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/timer/timer.hpp>

using namespace Vizia;

double test(DoomGame *dg, std::vector<int> *action, int iters, bool update){
    dg->init();

    boost::posix_time::ptime start = boost::posix_time::microsec_clock::local_time();
    boost::timer::cpu_timer timer;

    for(int i = 0;i<iters; ++i){
        if( dg->isEpisodeFinished() ){
            dg->newEpisode();
        }
        dg->setAction(action[rand()%3]);
        dg->advanceAction(1, update, update);
    }

    boost::timer::nanosecond_type cpuNsTime = timer.elapsed().wall; //timer.elapsed().system + timer.elapsed().user;

    boost::posix_time::ptime stop = boost::posix_time::microsec_clock::local_time();
    boost::posix_time::time_duration diff = stop - start;
    double posixTime = (double)diff.total_milliseconds()/1000;
    double boostTimer = (double)cpuNsTime/1000000000;
    std::cout<< posixTime << ",\t\t" << (double)iters/posixTime << ",\t\t" << boostTimer <<",\t\t" << (double)iters/boostTimer << "\n";

    dg->close();
    
    return posixTime;
}

int main(){

    DoomGame* dg= new DoomGame();

    dg->setDoomEnginePath("./viziazdoom");
    dg->setDoomGamePath("../scenarios/doom2.wad");
    dg->setDoomMap("map01");

    dg->setRenderHud(false);
    dg->setWindowVisible(false);

    dg->addAvailableButton(MOVE_LEFT);
    dg->addAvailableButton(MOVE_RIGHT);
    dg->addAvailableButton(ATTACK);

    dg->init();
    //dg->newEpisode();
    std::vector<int> action[3];

    action[0].push_back(1);
    action[0].push_back(0);
    action[0].push_back(0);

    action[1].push_back(0);
    action[1].push_back(1);
    action[1].push_back(0);

    action[2].push_back(0);
    action[2].push_back(0);
    action[2].push_back(1);

    int iters = 1000;
    double totalTime = 0;
    int testCount = 0;

    std::cout << "\nVIZIA FPS TEST\n";

    std::cout << "\nTICS: " <<iters <<"\n";

    std::cout << "\nSCENE: 20 ACTORS\n";
    dg->setDoomScenarioPath("../scenarios/20_actors.wad");
    dg->setScreenFormat(CRCGCB);
    std::cout << "\nTEST,\tRES,\t\tPOSIX_TIME,\t\tPOSIX_FPS,\t\tBOOST_TIME,\t\tBOOST_FPS\n";
    std::cout << ++testCount << ",\tlogic only,\t";
    totalTime += test(dg, action, iters, false);
    
    std::cout << ++testCount << ",\t160x120,\t";
    dg->setScreenResolution(RES_160X120);
    totalTime += test(dg, action, iters, true);
    std::cout << ++testCount << ",\t320x240,\t";
    dg->setScreenResolution(RES_320X240);
    totalTime += test(dg, action, iters, true);
    std::cout << ++testCount << ",\t640x480,\t";
    dg->setScreenResolution(RES_640X480);
    totalTime += test(dg, action, iters, true);
    std::cout << ++testCount << ",\t800x600,\t";
    dg->setScreenResolution(RES_800X600);
    totalTime += test(dg, action, iters, true);
    std::cout << ++testCount << ",\t1024x768,\t";
    dg->setScreenResolution(RES_1024X768);
    totalTime += test(dg, action, iters, true);

    dg->setScreenFormat(CRCGCBDB);
    std::cout << ++testCount << ",\t160x120+Z,\t";
    dg->setScreenResolution(RES_160X120);
    totalTime += test(dg, action, iters, true);
    std::cout << ++testCount << ",\t320x240+Z,\t";
    dg->setScreenResolution(RES_320X240);
    totalTime += test(dg, action, iters, true);
    std::cout << ++testCount << ",\t640x480+Z,\t";
    dg->setScreenResolution(RES_640X480);
    totalTime += test(dg, action, iters, true);
    std::cout << ++testCount << ",\t800x600+Z,\t";
    dg->setScreenResolution(RES_800X600);
    totalTime += test(dg, action, iters, true);
    std::cout << ++testCount << ",\t1024x768+Z,\t";
    dg->setScreenResolution(RES_1024X768);
    totalTime += test(dg, action, iters, true);

    std::cout << "\n\nSCENE: 50 ACTORS\n";
    std::cout << "\nTEST,\tRES,\t\tPOSIX_TIME,\t\tPOSIX_FPS,\t\tBOOST_TIME,\t\tBOOST_FPS\n";
    dg->setDoomScenarioPath("../scenarios/50_actors.wad");
    dg->setScreenFormat(CRCGCB);
    std::cout << ++testCount << ",\tlogic only,\t";
    totalTime += test(dg, action, iters, false);
    
    std::cout << ++testCount << ",\t160x120,\t";
    dg->setScreenResolution(RES_160X120);
    totalTime += test(dg, action, iters, true);
    std::cout << ++testCount << ",\t320x240,\t";
    dg->setScreenResolution(RES_320X240);
    totalTime += test(dg, action, iters, true);
    std::cout << ++testCount << ",\t640x480,\t";
    dg->setScreenResolution(RES_640X480);
    totalTime += test(dg, action, iters, true);
    std::cout << ++testCount << ",\t800x600,\t";
    dg->setScreenResolution(RES_800X600);
    totalTime += test(dg, action, iters, true);
    std::cout << ++testCount << ",\t1024x768,\t";
    dg->setScreenResolution(RES_1024X768);
    totalTime += test(dg, action, iters, true);

    dg->setScreenFormat(CRCGCBDB);
    std::cout << ++testCount << ",\t160x120+Z,\t";
    dg->setScreenResolution(RES_160X120);
    totalTime += test(dg, action, iters, true);
    std::cout << ++testCount << ",\t320x240+Z,\t";
    dg->setScreenResolution(RES_320X240);
    totalTime += test(dg, action, iters, true);
    std::cout << ++testCount << ",\t640x480+Z,\t";
    dg->setScreenResolution(RES_640X480);
    totalTime += test(dg, action, iters, true);
    std::cout << ++testCount << ",\t800x600+Z,\t";
    dg->setScreenResolution(RES_800X600);
    totalTime += test(dg, action, iters, true);
    std::cout << ++testCount << ",\t1024x768+Z,\t";
    dg->setScreenResolution(RES_1024X768);
    totalTime += test(dg, action, iters, true);

    std::cout << "\n\nSCENE: 100 ACTORS\n";
    std::cout << "\nTEST,\tRES,\t\tPOSIX_TIME,\t\tPOSIX_FPS,\t\tBOOST_TIME,\t\tBOOST_FPS\n";
    dg->setDoomScenarioPath("../scenarios/100_actors.wad");
    dg->setScreenFormat(CRCGCB);
    std::cout << ++testCount << ",\tlogic only,\t";
    totalTime += test(dg, action, iters, false);
    std::cout << ++testCount << ",\t160x120,\t";
    dg->setScreenResolution(RES_160X120);
    totalTime += test(dg, action, iters, true);
    std::cout << ++testCount << ",\t320x240,\t";
    dg->setScreenResolution(RES_320X240);
    totalTime += test(dg, action, iters, true);
    std::cout << ++testCount << ",\t640x480,\t";
    dg->setScreenResolution(RES_640X480);
    totalTime += test(dg, action, iters, true);
    std::cout << ++testCount << ",\t800x600,\t";
    dg->setScreenResolution(RES_800X600);
    totalTime += test(dg, action, iters, true);
    std::cout << ++testCount << ",\t1024x768,\t";
    dg->setScreenResolution(RES_1024X768);
    totalTime += test(dg, action, iters, true);

    dg->setScreenFormat(CRCGCBDB);
    std::cout << ++testCount << ",\t160x120+Z,\t";
    dg->setScreenResolution(RES_160X120);
    totalTime += test(dg, action, iters, true);
    std::cout << ++testCount << ",\t320x240+Z,\t";
    dg->setScreenResolution(RES_320X240);
    totalTime += test(dg, action, iters, true);
    std::cout << ++testCount << ",\t640x480+Z,\t";
    dg->setScreenResolution(RES_640X480);
    totalTime += test(dg, action, iters, true);
    std::cout << ++testCount << ",\t800x600+Z,\t";
    dg->setScreenResolution(RES_800X600);
    totalTime += test(dg, action, iters, true);
    std::cout << ++testCount << ",\t1024x768+Z,\t";
    dg->setScreenResolution(RES_1024X768);
    totalTime += test(dg, action, iters, true);

    std::cout << "\n\nSCENE: 300 ACTORS\n";
    std::cout << "\nTEST,\tRES,\t\tPOSIX_TIME,\t\tPOSIX_FPS,\t\tBOOST_TIME,\t\tBOOST_FPS\n";
    dg->setDoomScenarioPath("../scenarios/550_actors.wad");
    dg->setScreenFormat(CRCGCB);
    std::cout << ++testCount << ",\tlogic only,\t";
    totalTime += test(dg, action, iters, false);
    std::cout << ++testCount << ",\t160x120,\t";
    dg->setScreenResolution(RES_160X120);
    totalTime += test(dg, action, iters, true);
    std::cout << ++testCount << ",\t320x240,\t";
    dg->setScreenResolution(RES_320X240);
    totalTime += test(dg, action, iters, true);
    std::cout << ++testCount << ",\t640x480,\t";
    dg->setScreenResolution(RES_640X480);
    totalTime += test(dg, action, iters, true);
    std::cout << ++testCount << ",\t800x600,\t";
    dg->setScreenResolution(RES_800X600);
    totalTime += test(dg, action, iters, true);
    std::cout << ++testCount << ",\t1024x768,\t";
    dg->setScreenResolution(RES_1024X768);
    totalTime += test(dg, action, iters, true);

    dg->setScreenFormat(CRCGCBDB);
    std::cout << ++testCount << ",\t160x120+Z,\t";
    dg->setScreenResolution(RES_160X120);
    totalTime += test(dg, action, iters, true);
    std::cout << ++testCount << ",\t320x240+Z,\t";
    dg->setScreenResolution(RES_320X240);
    totalTime += test(dg, action, iters, true);
    std::cout << ++testCount << ",\t640x480+Z,\t";
    dg->setScreenResolution(RES_640X480);
    totalTime += test(dg, action, iters, true);
    std::cout << ++testCount << ",\t800x600+Z,\t";
    dg->setScreenResolution(RES_800X600);
    totalTime += test(dg, action, iters, true);
    std::cout << ++testCount << ",\t1024x768+Z,\t";
    dg->setScreenResolution(RES_1024X768);
    totalTime += test(dg, action, iters, true);

    std::cout << "\n\nSCENE: 500 ACTORS\n";
    std::cout << "\nTEST,\tRES,\t\tPOSIX_TIME,\t\tPOSIX_FPS,\t\tBOOST_TIME,\t\tBOOST_FPS\n";
    dg->setDoomScenarioPath("../scenarios/500_actors.wad");
    dg->setScreenFormat(CRCGCB);
    std::cout << ++testCount << ",\tlogic only,\t";
    totalTime += test(dg, action, iters, false);
    
    std::cout << ++testCount << ",\t160x120,\t";
    dg->setScreenResolution(RES_160X120);
    totalTime += test(dg, action, iters, true);
    std::cout << ++testCount << ",\t320x240,\t";
    dg->setScreenResolution(RES_320X240);
    totalTime += test(dg, action, iters, true);
    std::cout << ++testCount << ",\t640x480,\t";
    dg->setScreenResolution(RES_640X480);
    totalTime += test(dg, action, iters, true);
    std::cout << ++testCount << ",\t800x600,\t";
    dg->setScreenResolution(RES_800X600);
    totalTime += test(dg, action, iters, true);
    std::cout << ++testCount << ",\t1024x768,\t";
    dg->setScreenResolution(RES_1024X768);
    totalTime += test(dg, action, iters, true);

    dg->setScreenFormat(CRCGCBDB);
    std::cout << ++testCount << ",\t160x120+Z,\t";
    dg->setScreenResolution(RES_160X120);
    totalTime += test(dg, action, iters, true);
    std::cout << ++testCount << ",\t320x240+Z,\t";
    dg->setScreenResolution(RES_320X240);
    totalTime += test(dg, action, iters, true);
    std::cout << ++testCount << ",\t640x480+Z,\t";
    dg->setScreenResolution(RES_640X480);
    totalTime += test(dg, action, iters, true);
    std::cout << ++testCount << ",\t800x600+Z,\t";
    dg->setScreenResolution(RES_800X600);
    totalTime += test(dg, action, iters, true);
    std::cout << ++testCount << ",\t1024x768+Z,\t";
    dg->setScreenResolution(RES_1024X768);
    totalTime += test(dg, action, iters, true);

    std::cout << "\nTOTAL TIME: " << totalTime << " s\n";

    delete dg;
}
