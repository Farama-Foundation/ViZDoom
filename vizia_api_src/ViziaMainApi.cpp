#include "ViziaMain.h"
#include <iostream>
#include <vector>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>
#include <boost/fusion/container/list.hpp>
#include <boost/fusion/include/list.hpp>
#include <boost/fusion/container/list/list_fwd.hpp>
#include <boost/fusion/include/list_fwd.hpp>


class ViziaMainApi{
		
    public:
        ViziaMainApi(){
            this->main=new ViziaMain();
            import_array();
        }
        ~ViziaMainApi(){delete(this->main);}

        void loadConfig(std::string file){this->main->loadConfig(file);}

        void init(){this->main->init();}
        void close(){this->main->close();}

        void newEpisode(){this->main->newEpisode();}
        
	float makeAction(boost::python::list _list)
	{
		int list_length = boost::python::len(_list);
		std::vector<bool> action = std::vector<bool>(list_length);
		for (int i=0; i<list_length; i++)
		{
			action[i]=boost::python::extract<bool>(_list[i]);
		}
		return this->main->makeAction(action);
	}

        ViziaMain::State getState(){return this->main->getState();}
        bool * getLastActions(){return this->main-> getLastActions();}
        bool isNewEpisode(){return this->main->isNewEpisode();}
        bool isEpisodeFinished(){ return this->main->isEpisodeFinished();}
        void addAvailableAction(int action){this->main->addAvailableAction(action);}
        void addAvailableAction(std::string action){this->main->addAvailableAction(action);}

        void addStateAvailableVar(int var){this->main->addStateAvailableVar(var);}
        void addStateAvailableVar(std::string var){this->main->addStateAvailableVar(var);}

        //OPTIONS

        const ViziaDoomController* getController(){return this->main->getController();}

        void setDoomGamePath(std::string path){this->main->setDoomGamePath(path);}
        void setDoomIwadPath(std::string path){this->main->setDoomIwadPath(path);}
        void setDoomFilePath(std::string path){this->main->setDoomFilePath(path);}
        void setDoomMap(std::string map){this->main->setDoomMap(map);}
        void setDoomSkill(int skill){this->main->setDoomSkill(skill);}
        void setDoomConfigPath(std::string path){this->main->setDoomConfigPath(path);}

        void setNewEpisodeOnTimeout(bool set){this->main->setNewEpisodeOnTimeout(set);}
        void setNewEpisodeOnPlayerDeath(bool set){this->main->setNewEpisodeOnPlayerDeath(set);}
        void setEpisodeTimeoutInMiliseconds(unsigned int ms){this->main->setEpisodeTimeoutInMiliseconds(ms);}
        void setEpisodeTimeoutInDoomTics(unsigned int tics){this->main->setEpisodeTimeoutInDoomTics(tics);}

        void setScreenResolution(int width, int height){this->main->setScreenResolution(width,height);}
        void setScreenWidth(int width){this->main->setScreenWidth(width);}
        void setScreenHeight(int height){this->main->setScreenHeight(height);}
        void setScreenFormat(int format){this->main->setScreenFormat(format);}
        void setRenderHud(bool hud){this->main->setRenderHud(hud);}
        void setRenderWeapon(bool weapon){this->main->setRenderWeapon(weapon);}
        void setRenderCrosshair(bool crosshair){this->main->setRenderCrosshair(crosshair);}
        void setRenderDecals(bool decals){this->main->setRenderDecals(decals);}
        void setRenderParticles(bool particles){this->main->setRenderParticles(particles);}

        int getScreenWidth(){return this->main->getScreenWidth();}
        int getScreenHeight(){return this->main->getScreenHeight();}
        int getScreenPitch(){return this->main->getScreenPitch();}
        int getScreenSize(){return this->main->getScreenSize();}
        int getScreenFormat(){return this->main->getScreenFormat();}

    private:

        ViziaMain * main;

        std::vector<int> stateAvailableVars;
        std::vector<int> availableActions;

        int *stateVars;
        bool *lastActions;

};
