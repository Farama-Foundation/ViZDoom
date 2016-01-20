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
                case CRCGCBZB:
                case CBCGCR:
                case CBCGCRZB:
                case GRAY8:
                case ZBUFFER8:
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

    void DoomGamePython::setAction(boost::python::list &action) {
        // TODO what if isFinished()?
        int listLength = boost::python::len(action);
        if( listLength != this->getAvailableButtonsSize())
        {
            cerr<<"Incorrect action length: "<<listLength<<" Should be: "<<this->getAvailableButtonsSize()<<endl;
            //TODO maybe throw something?
            return ;
        }
        std::vector<int> properAction = std::vector<int>(listLength);
        for (int i = 0; i < listLength; i++) {
            properAction[i] = boost::python::extract<int>(action[i]);
        }
        DoomGame::setAction(properAction);
        
    }

    double DoomGamePython::makeAction(boost::python::list &action)
    {
        this->setAction(action);
        DoomGame::advanceAction();
        return DoomGame::getLastReward();
    }

    double DoomGamePython::makeAction(boost::python::list &action, unsigned int tics)
    {
        this->setAction(action);
        DoomGame::advanceAction(true, true, tics);
        return DoomGame::getLastReward();
    }

    DoomGamePython::PythonState DoomGamePython::getState() {
        if (isEpisodeFinished()) {
            return DoomGamePython::PythonState(this->state.number);
        }

        PyObject *img = PyArray_SimpleNewFromData(3, imageShape, NPY_UBYTE, this->doomController->getScreen());
        boost::python::handle<> numpyImageHandle = boost::python::handle<>(img);
        boost::python::numeric::array numpyImage = array(numpyImageHandle);

        if (this->state.gameVariables.size() > 0) {
            npy_intp varLen = this->state.gameVariables.size();
            PyObject *vars = PyArray_SimpleNewFromData(1, &varLen, NPY_INT32, this->state.gameVariables.data());
            boost::python::handle<> numpyVarsHandle = boost::python::handle<>(vars);
            boost::python::numeric::array numpyVars = array(numpyVarsHandle);

            return DoomGamePython::PythonState(state.number, numpyImage.copy(), numpyVars.copy());
        }
        else {
            return DoomGamePython::PythonState(state.number, numpyImage.copy());
        }


    }

    boost::python::list DoomGamePython::getLastAction() {
        boost::python::list res;
        std::vector<int> lastAction = DoomGame::getLastAction();
        for (std::vector<int>::iterator it = lastAction.begin(); it!=lastAction.end();++it)
        {
            res.append(*it); 
        }
        return res;
    }

    object DoomGamePython::getGameScreen(){
        //TODO check if it works
        PyObject *img = PyArray_SimpleNewFromData(3, imageShape, NPY_UBYTE, this->doomController->getScreen());
        boost::python::handle<> numpyImageHandle = boost::python::handle<>(img);
        boost::python::numeric::array numpyImage = array(numpyImageHandle);
        return numpyImage.copy();
    }

}