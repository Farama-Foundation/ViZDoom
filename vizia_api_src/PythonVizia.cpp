#include <boost/python.hpp>
#include "ViziaPythonApi.cpp"
#include "ViziaMain.cpp"
#include "ViziaDoomController.cpp"

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

void    (ViziaPythonApi::*addAvailableKeyInt)(int)              = &ViziaPythonApi::addAvailableKey;
void    (ViziaPythonApi::*addAvailableKeyString)(std::string)      = &ViziaPythonApi::addAvailableKey;
void    (ViziaPythonApi::*addStateAvailableVarInt)(int)              = &ViziaPythonApi::addStateAvailableVar;
void    (ViziaPythonApi::*addStateAvailableVarString)(std::string)      = &ViziaPythonApi::addStateAvailableVar;


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

	.def("addStateAvailableVar", addStateAvailableVarInt)
	.def("addStateAvailableVar", addStateAvailableVarString)
	.def("addAvailableKey", addAvailableKeyInt)
	.def("addAvailableKey", addAvailableKeyString)
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
