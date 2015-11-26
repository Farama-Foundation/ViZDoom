#include <boost/python.hpp>
#include "ViziaMainPython.h"
#include "ViziaDefines.h"
/*C++ code to expose ViziaMainPython via python */
using namespace boost::python;

BOOST_PYTHON_MODULE(vizia)
{
	/*I don't know if it's needed or not */
	//Py_Initialize();

	enum_<ViziaScreenFormat>("ScreenFormat")
		.value("CRCGCB", CRCGCB)
		.value("CRCGCBCA", CRCGCBCA)
		.value("RGB24", RGB24)
		.value("RGBA32", RGBA32)
		.value("ARGB32", ARGB32)
		.value("CBCGCR", CBCGCR)
		.value("CBCGCRCA", CBCGCRCA)
		.value("BGR24", BGR24)
		.value("BGRA32", BGRA32)
		.value("ABGR32", ABGR32)
		;

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

    class_<ViziaMainPython>("ViziaGame", init<>())
		.def("init", &ViziaMainPython::init)
		.def("loadConfig", &ViziaMainPython::loadConfig)
		.def("close", &ViziaMainPython::close)
		.def("newEpisode", &ViziaMainPython::newEpisode)
		.def("isEpisodeFinished", &ViziaMainPython::isEpisodeFinished)
		.def("isNewEpisode", &ViziaMainPython::isNewEpisode)
		.def("makeAction",&ViziaMainPython::makeAction)
		.def("getState", &ViziaMainPython::getState)
		.def("getStateFormat", &ViziaMainPython::getStateFormat)
		.def("getActionFormat", &ViziaMainPython::getActionFormat)

		.def("addStateAvailableVar", &ViziaMainPython::addStateAvailableVar)
		.def("addAvailableButton", &ViziaMainPython::addAvailableButton)
		.def("setDoomGamePath", &ViziaMainPython::setDoomGamePath)
		.def("setDoomIwadPath", &ViziaMainPython::setDoomIwadPath)
		.def("setDoomFilePath", &ViziaMainPython::setDoomFilePath)
		.def("setDoomMap", &ViziaMainPython::setDoomMap)
		.def("setDoomSkill", &ViziaMainPython::setDoomSkill)
		.def("setDoomConfigPath", &ViziaMainPython::setDoomConfigPath)
		.def("setNewEpisodeOnTimeout", &ViziaMainPython::setNewEpisodeOnTimeout)
		.def("setNewEpisodeOnPlayerDeath", &ViziaMainPython::setNewEpisodeOnPlayerDeath)
		.def("setEpisodeTimeoutInMiliseconds", &ViziaMainPython::setEpisodeTimeoutInMiliseconds)
		.def("setEpisodeTimeoutInDoomTics", &ViziaMainPython::setEpisodeTimeoutInDoomTics)
		.def("setScreenResolution", &ViziaMainPython::setScreenResolution)
		.def("setScreenWidth", &ViziaMainPython::setScreenWidth)
		.def("setScreenHeight", &ViziaMainPython::setScreenHeight)
		.def("setScreenFormat", &ViziaMainPython::setScreenFormat)
		.def("setRenderHud", &ViziaMainPython::setRenderHud)
		.def("setRenderWeapon", &ViziaMainPython::setRenderWeapon)
		.def("setRenderCrosshair", &ViziaMainPython::setRenderCrosshair)
		.def("setRenderDecals", &ViziaMainPython::setRenderDecals)
		.def("setRenderParticles", &ViziaMainPython::setRenderParticles)
		.def("getScreenWidth", &ViziaMainPython::getScreenWidth)
		.def("getScreenHeight", &ViziaMainPython::getScreenHeight)
		.def("getScreenSize", &ViziaMainPython::getScreenSize)
		.def("getScreenPitch", &ViziaMainPython::getScreenPitch)
		.def("getScreenFormat", &ViziaMainPython::getScreenFormat)
	
    ;

}
