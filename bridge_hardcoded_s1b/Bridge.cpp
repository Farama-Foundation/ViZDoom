#include "ViziaDoomController.h"
#include <iostream>
#include "Bridge.h"


VIZIA_StateFormat* _state_format;
VIZIA_State* _state;
ViziaDoomController *vdm;
	
int _killed; //previous amount of Player's kills
double _summary_reward;
int _finished;
	
void init(int x, int y, int maxtime) //
{
	
	vdm = new ViziaDoomController; 
    std::cout << "SETTING DOOM " << std::endl;

    vdm->setGamePath("zdoom");
    vdm->setIwadPath("dooom2.wad");
    vdm->setFilePath("s1_b.wad");
    vdm->setMap("map01");
	vdm->setMapTimeout(300);
	vdm->setScreenSize(x, y);
	
	if (maxtime>0){
    	vdm->setMapTimeout(maxtime);
	}

    vdm->showHud(false);
    vdm->showCrosshair(true);
    vdm->showWeapon(true);
    vdm->showDecals(false);
    vdm->showParticles(false);
	
	_summary_reward=0;
	_killed=0;	
	_finished=0;
	
	_state_format = new VIZIA_StateFormat();
	_state = new VIZIA_State();
	_state_format->image_shape_len = 2;
	_state_format->image_shape = new int[2];
	_state_format->image_shape[0] = x;
	_state_format->image_shape[1] = y;
	
    vdm->init();
	
	for (int i=0;i<15;i++){
	vdm->tic();
	}

}
void new_episode()  //
{

	vdm->restartMap();
	vdm->getPlayerKillCount(); //just because, it isnt working without it.
	//vdm->init();
	
	for (int i=0;i<15;i++){
	vdm->tic();
	}
	
	
	_finished=0;
	_summary_reward=0;
	_killed=0;
}
int get_action_format() //
{
	return 3;
}
VIZIA_StateFormat* get_state_format() //
{
	return _state_format;
}

VIZIA_State* get_state() 
{
	return _state;
}
double make_action(int const* action)
{

	vdm->setButtonState(V_MOVELEFT, action[0]);
	vdm->setButtonState(V_MOVERIGHT, action[1]);
	vdm->setButtonState(V_ATTACK, action[2]);
	
	vdm->tic();
	double reward=-0.001;
	if (vdm->getPlayerKillCount()>_killed)
	{
		reward=reward+1;
		_finished=1;
		
	}
	_killed=vdm->getPlayerKillCount();
	_summary_reward+=reward;
	return reward;
	
}


double get_summary_reward(){return _summary_reward;}
int is_finished(){return _finished;}
void close(){
	vdm->close();
	_finished=1;
	

}
