#include <boost/python.hpp>
#include "ViziaPythonApi.cpp"
#include "ViziaMain.cpp"
#include "ViziaDoomController.cpp"
#include "ViziaDefines.h"
/*C++ code to expose ViziaPythonApi via python */
using namespace boost::python;



class Py_ViziaMain{
public:
static ViziaMain* init(){
    return new ViziaMain();
}
};


BOOST_PYTHON_MODULE(vizia)
{

	enum_<ViziaButton>("Button")
        .value("ATTACK", ATTACK)
        .value("USE", USE)
        .value("JUMP", JUMP)
        .value("CROUCH", CROUCH)
        .value("TURN180", TURN180)
        .value("ALTATTACK", ALTATTACK)
        .value("RELOAD", RELOAD)
        .value("ZOOM", ZOOM)
        .value("SPEED", SPEED)
        .value("STRAFE", STRAFE)
        .value("MOVERIGHT", MOVERIGHT)
        .value("MOVELEFT", MOVELEFT)
        .value("BACK", BACK)
        .value("FORWARD", FORWARD)
        .value("RIGHT", RIGHT)
        .value("LEFT", LEFT)
        .value("LOOKUP", LOOKUP)
        .value("LOOKDOWN", LOOKDOWN)
        .value("SELECT_WEAPON1", SELECT_WEAPON1)
        .value("SELECT_WEAPON2", SELECT_WEAPON2)
        .value("SELECT_WEAPON3", SELECT_WEAPON3)
        .value("SELECT_WEAPON4", SELECT_WEAPON4)
        .value("SELECT_WEAPON5", SELECT_WEAPON5)
        .value("SELECT_WEAPON6", SELECT_WEAPON6)
        .value("SELECT_WEAPON7", SELECT_WEAPON7)
        .value("SELECT_NEXT_WEAPON", SELECT_NEXT_WEAPON)
        .value("SELECT_PREV_WEAPON", SELECT_PREV_WEAPON)
        .value("UNDEFINED_BUTTON", UNDEFINED_BUTTON)
        ;

    enum_<ViziaGameVar>("GameVar")
        .value("KILLCOUNT", KILLCOUNT)
        .value("ITEMCOUNT", ITEMCOUNT)
        .value("SECRETCOUNT", SECRETCOUNT)
        .value("HEALTH", HEALTH)
        .value("ARMOR", ARMOR)
        .value("SELECTED_WEAPON", SELECTED_WEAPON)
        .value("SELECTED_WEAPON_AMMO", SELECTED_WEAPON_AMMO)
        .value("AMMO1", AMMO1)
        .value("AMMO2", AMMO2)
        .value("AMMO3", AMMO3)
        .value("AMMO4", AMMO4)
        .value("WEAPON1", WEAPON1)
        .value("WEAPON2", WEAPON2)
        .value("WEAPON3", WEAPON3)
        .value("WEAPON4", WEAPON4)
        .value("WEAPON5", WEAPON5)
        .value("WEAPON6", WEAPON6)
        .value("WEAPON7", WEAPON7)
        .value("KEY1", KEY1)
        .value("KEY2", KEY2)
        .value("KEY3", KEY3)
        .value("UNDEFINED_VAR", UNDEFINED_VAR)
        ;
    

	def("DoomTics2Ms", DoomTics2Ms);
	def("Ms2DoomTics", Ms2DoomTics);

    class_<ViziaPythonApi>("ViziaGame", init<>())
	.def("init", &ViziaPythonApi::init)
	.def("loadConfig", &ViziaPythonApi::loadConfig)
	.def("close", &ViziaPythonApi::close)
	.def("newEpisode", &ViziaPythonApi::newEpisode)
	.def("isEpisodeFinished", &ViziaPythonApi::isEpisodeFinished)
	.def("isNewEpisode", &ViziaPythonApi::isNewEpisode)
	.def("makeAction",&ViziaPythonApi::makeAction)
	.def("getState", &ViziaPythonApi::getState)
	.def("getStateFormat", &ViziaPythonApi::getStateFormat)
	.def("getActionFormat", &ViziaPythonApi::getActionFormat)

	.def("addStateAvailableVar", &ViziaPythonApi::addStateAvailableVar)
	.def("addAvailableButton", &ViziaPythonApi::addAvailableButton)
	.def("setDoomGamePath", &ViziaPythonApi::setDoomGamePath)
	.def("setDoomIwadPath", &ViziaPythonApi::setDoomIwadPath)
	.def("setDoomFilePath", &ViziaPythonApi::setDoomFilePath)
	.def("setDoomMap", &ViziaPythonApi::setDoomMap)
	.def("setDoomSkill", &ViziaPythonApi::setDoomSkill)
	.def("setDoomConfigPath", &ViziaPythonApi::setDoomConfigPath)
	.def("setNewEpisodeOnTimeout", &ViziaPythonApi::setNewEpisodeOnTimeout)
	.def("setNewEpisodeOnPlayerDeath", &ViziaPythonApi::setNewEpisodeOnPlayerDeath)
	.def("setEpisodeTimeoutInMiliseconds", &ViziaPythonApi::setEpisodeTimeoutInMiliseconds)
	.def("setEpisodeTimeoutInDoomTics", &ViziaPythonApi::setEpisodeTimeoutInDoomTics)
	.def("setScreenResolution", &ViziaPythonApi::setScreenResolution)
	.def("setScreenWidth", &ViziaPythonApi::setScreenWidth)
	.def("setScreenHeight", &ViziaPythonApi::setScreenHeight)
	.def("setScreenFormat", &ViziaPythonApi::setScreenFormat)
	.def("setRenderHud", &ViziaPythonApi::setRenderHud)
	.def("setRenderWeapon", &ViziaPythonApi::setRenderWeapon)
	.def("setRenderCrosshair", &ViziaPythonApi::setRenderCrosshair)
	.def("setRenderDecals", &ViziaPythonApi::setRenderDecals)
	.def("setRenderParticles", &ViziaPythonApi::setRenderParticles)
	.def("getScreenWidth", &ViziaPythonApi::getScreenWidth)
	.def("getScreenHeight", &ViziaPythonApi::getScreenHeight)
	.def("getScreenSize", &ViziaPythonApi::getScreenSize)
	.def("getScreenPitch", &ViziaPythonApi::getScreenPitch)
	.def("getScreenFormat", &ViziaPythonApi::getScreenFormat)
	
    ;

}
