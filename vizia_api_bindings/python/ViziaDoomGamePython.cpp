#include "ViziaDoomGamePython.h"

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

    std::vector<int> DoomGamePython::pyListToIntVector(boost::python::list const &action)
    {
        int listLength = boost::python::len(action);
        std::vector<int> properAction = std::vector<int>(listLength);
        for (int i = 0; i < listLength; i++) {
            properAction[i] = boost::python::extract<int>(action[i]);
        }
        return properAction;
    }
    void DoomGamePython::setAction(boost::python::list const &action) {
        DoomGame::setAction(DoomGamePython::pyListToIntVector(action));
    }

    double DoomGamePython::makeAction(boost::python::list const &action)
    {
        return DoomGame::makeAction(DoomGamePython::pyListToIntVector(action));
    }

    double DoomGamePython::makeAction(boost::python::list const &action, unsigned int tics)
    {
        return DoomGame::makeAction(DoomGamePython::pyListToIntVector(action), tics);   
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
        for (std::vector<int>::iterator it = DoomGame::lastAction.begin(); it!=DoomGame::lastAction.end(); ++it)
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