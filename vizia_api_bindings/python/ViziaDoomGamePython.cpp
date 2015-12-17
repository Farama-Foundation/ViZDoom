#include "ViziaDoomGamePython.h"
#include <iostream>
using std::cout;
using std::cerr;
using std::endl;
namespace Vizia {

    using boost::python::tuple;
    using boost::python::api::object;
    using boost::python::numeric::array;
#define PY_NONE object()

    DoomGamePython::DoomGamePython() {
        boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
        import_array(); 
    }


    bool DoomGamePython::init() {
        bool initSuccess = DoomGame::init();

        if (initSuccess) {

            int channels = this->getScreenChannels();
            int x = this->getScreenWidth();
            int y = this->getScreenHeight();
            
            switch(this->getScreenFormat())
            {
                case CRCGCB:
                case CRCGCBCA:
                case CBCGCR:
                case CBCGCRCA:
                case GRAY8:
                    this->imageShape[0] = channels;
                    this->imageShape[1] = y;
                    this->imageShape[2] = x;
                    break;
                default:
                    this->imageShape[0] = y;
                    this->imageShape[1] = x;
                    this->imageShape[2] = channels;
            }
            

        }
        return initSuccess;
    }

    float DoomGamePython::makeAction(boost::python::list actionList) {
        // TODO what if isFinished()?
        int listLength = boost::python::len(actionList);
        if( listLength != this->getActionFormat())
        {
            cerr<<"Incorrect action length: "<<listLength<<" Should be: "<<this->getActionFormat()<<endl;
            //maybe throw something?
            return 0;
        }
        std::vector<bool> action = std::vector<bool>(listLength);
        for (int i = 0; i < listLength; i++) {
            action[i] = boost::python::extract<bool>(actionList[i]);
        }
        return DoomGame::makeAction(action);
        
    }

    DoomGamePython::PythonState DoomGamePython::getState() {
        if (isEpisodeFinished()) {
            return DoomGamePython::PythonState(this->state.number);
        }


        PyObject *img = PyArray_SimpleNewFromData(3, imageShape, NPY_UBYTE, this->doomController->getScreen());
        boost::python::handle<> numpyImageHandle = boost::python::handle<>(img);
        boost::python::numeric::array numpyImage = array(numpyImageHandle);

        if (this->state.vars.size() > 0) {
            npy_intp varLen = this->state.vars.size();
            PyObject *vars = PyArray_SimpleNewFromData(1, &varLen, NPY_INT32, this->state.vars.data());
            boost::python::handle<> numpyVarsHandle = boost::python::handle<>(vars);
            boost::python::numeric::array numpyVars = array(numpyVarsHandle);

            return DoomGamePython::PythonState(state.number, numpyImage.copy(), numpyVars.copy());
        }
        else {
            return DoomGamePython::PythonState(state.number, numpyImage.copy());
        }


    }

/* not sure if we need this */
    boost::python::list DoomGamePython::getLastAction() {
        boost::python::list res;
        std::vector<bool> lastAction = DoomGame::getLastAction();
        for (std::vector<bool>::iterator it = lastAction.begin(); it!=lastAction.end();++it)
        {
            //TODO
            //insert *it somehow
            //res.append(*it); <- this doesn't work and I have other stuff to do
        }
        return res;
    }
}