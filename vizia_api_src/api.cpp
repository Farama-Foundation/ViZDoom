#include <boost/python.hpp>
#include "ViziaMain.cpp"
#include "ViziaDoomController.cpp"
using namespace boost::python;

BOOST_PYTHON_MODULE(api)
{
void    (ViziaMain::*addAvailableActionInt)(int)              = &ViziaMain::addAvailableAction;
void    (ViziaMain::*addAvailableActionString)(std::string)      = &ViziaMain::addAvailableAction;
void    (ViziaMain::*addStateAvailableVarInt)(int)              = &ViziaMain::addStateAvailableVar;
void    (ViziaMain::*addStateAvailableVarString)(std::string)      = &ViziaMain::addStateAvailableVar;
def("DoomTics2Ms", DoomTics2Ms);
def("Ms2DoomTics", Ms2DoomTics);

    class_<ViziaMain>("ViziaMain", init<>())
	.def("init", &ViziaMain::init)
	.def("loadConfig", &ViziaMain::loadConfig)
	.def("close", &ViziaMain::close)
	.def("newEpisode", &ViziaMain::newEpisode)
	//.def("getLastActions", &ViziaMain::getLastActions)
	.def("addStateAvailableVar", addStateAvailableVarInt)
	.def("addStateAvailableVar", addStateAvailableVarString)
	.def("isNewEpisode", &ViziaMain::isNewEpisode)
	.def("addAvailableAction", addAvailableActionInt)
	.def("addAvailableAction", addAvailableActionString)
	.def("setDoomGamePath", &ViziaMain::setDoomGamePath)
	.def("setDoomIwadPath", &ViziaMain::setDoomIwadPath)
	.def("setDoomFilePath", &ViziaMain::setDoomFilePath)
	.def("setDoomMap", &ViziaMain::setDoomMap)
	.def("setDoomSkill", &ViziaMain::setDoomSkill)
	.def("setDoomConfigPath", &ViziaMain::setDoomConfigPath)
	.def("setNewEpisodeOnTimeout", &ViziaMain::setNewEpisodeOnTimeout)
	.def("setNewEpisodeOnPlayerDeath", &ViziaMain::setNewEpisodeOnPlayerDeath)
	.def("setEpisodeTimeoutInMiliseconds", &ViziaMain::setEpisodeTimeoutInMiliseconds)
	.def("setEpisodeTimeoutInDoomTics", &ViziaMain::setEpisodeTimeoutInDoomTics)
	.def("setScreenResolution", &ViziaMain::setScreenResolution)
	.def("setScreenWidth", &ViziaMain::setScreenWidth)
	.def("setScreenHeight", &ViziaMain::setScreenHeight)
	.def("setScreenFormat", &ViziaMain::setScreenFormat)
	.def("setRenderHud", &ViziaMain::setRenderHud)
	.def("setRenderWeapon", &ViziaMain::setRenderWeapon)
	.def("setRenderCrosshair", &ViziaMain::setRenderCrosshair)
	.def("setRenderDecals", &ViziaMain::setRenderDecals)
	.def("setRenderParticles", &ViziaMain::setRenderParticles)
	.def("getScreenWidth", &ViziaMain::getScreenWidth)
	.def("getScreenHeight", &ViziaMain::getScreenHeight)
	.def("getScreenSize", &ViziaMain::getScreenSize)
	.def("getScreenPitch", &ViziaMain::getScreenPitch)
	.def("getScreenFormat", &ViziaMain::getScreenFormat)

    ;

}
