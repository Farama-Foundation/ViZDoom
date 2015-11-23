#include <boost/python.hpp>
#include "ViziaMainApi.cpp"
#include "ViziaMain.cpp"
#include "ViziaDoomController.cpp"
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>

using namespace boost::python;
class ViziaMain2{
public:
static ViziaMain* init(){
    return new ViziaMain();
}
};


BOOST_PYTHON_MODULE(api)
{

void    (ViziaMainApi::*addAvailableActionInt)(int)              = &ViziaMainApi::addAvailableAction;
void    (ViziaMainApi::*addAvailableActionString)(std::string)      = &ViziaMainApi::addAvailableAction;
void    (ViziaMainApi::*addStateAvailableVarInt)(int)              = &ViziaMainApi::addStateAvailableVar;
void    (ViziaMainApi::*addStateAvailableVarString)(std::string)      = &ViziaMainApi::addStateAvailableVar;


def("DoomTics2Ms", DoomTics2Ms);
def("Ms2DoomTics", Ms2DoomTics);

    class_<ViziaMainApi>("ViziaMain", init<>())
	.def("init", &ViziaMainApi::init)
	.def("loadConfig", &ViziaMainApi::loadConfig)
	.def("close", &ViziaMainApi::close)
	.def("newEpisode", &ViziaMainApi::newEpisode)
	//.def("getLastActions", &ViziaMain::getLastActions)
	.def("addStateAvailableVar", addStateAvailableVarInt)
	.def("addStateAvailableVar", addStateAvailableVarString)
	.def("isNewEpisode", &ViziaMainApi::isNewEpisode)
	.def("addAvailableAction", addAvailableActionInt)
	.def("addAvailableAction", addAvailableActionString)
	.def("setDoomGamePath", &ViziaMainApi::setDoomGamePath)
	.def("setDoomIwadPath", &ViziaMainApi::setDoomIwadPath)
	.def("setDoomFilePath", &ViziaMainApi::setDoomFilePath)
	.def("setDoomMap", &ViziaMainApi::setDoomMap)
	.def("setDoomSkill", &ViziaMainApi::setDoomSkill)
	.def("setDoomConfigPath", &ViziaMainApi::setDoomConfigPath)
	.def("setNewEpisodeOnTimeout", &ViziaMainApi::setNewEpisodeOnTimeout)
	.def("setNewEpisodeOnPlayerDeath", &ViziaMainApi::setNewEpisodeOnPlayerDeath)
	.def("setEpisodeTimeoutInMiliseconds", &ViziaMainApi::setEpisodeTimeoutInMiliseconds)
	.def("setEpisodeTimeoutInDoomTics", &ViziaMainApi::setEpisodeTimeoutInDoomTics)
	.def("setScreenResolution", &ViziaMainApi::setScreenResolution)
	.def("setScreenWidth", &ViziaMainApi::setScreenWidth)
	.def("setScreenHeight", &ViziaMainApi::setScreenHeight)
	.def("setScreenFormat", &ViziaMainApi::setScreenFormat)
	.def("setRenderHud", &ViziaMainApi::setRenderHud)
	.def("setRenderWeapon", &ViziaMainApi::setRenderWeapon)
	.def("setRenderCrosshair", &ViziaMainApi::setRenderCrosshair)
	.def("setRenderDecals", &ViziaMainApi::setRenderDecals)
	.def("setRenderParticles", &ViziaMainApi::setRenderParticles)
	.def("getScreenWidth", &ViziaMainApi::getScreenWidth)
	.def("getScreenHeight", &ViziaMainApi::getScreenHeight)
	.def("getScreenSize", &ViziaMainApi::getScreenSize)
	.def("getScreenPitch", &ViziaMainApi::getScreenPitch)
	.def("getScreenFormat", &ViziaMainApi::getScreenFormat)
	.def("makeAction",&ViziaMainApi::makeAction)
    ;

}
