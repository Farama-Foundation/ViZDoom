#include "ViziaDoomGamePython.h"

#include <iostream>
using std::cout;
using std::endl;

namespace Vizia {

    using boost::python::tuple;
    using boost::python::api::object;

#define PY_NONE object()

    DoomGamePython::DoomGamePython() {
        import_array();
        Py_Initialize();
        /* Numpy arrays won't work unless this strnage function is envoked.*/
        boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
        //boost::numpy::initialize();
    }

    bool DoomGamePython::init() {
        bool initSuccess = DoomGame::init();
        /* fill state format */
        if (initSuccess) {
            

            DoomGame::StateFormat cppFormat = DoomGame::getStateFormat();
            boost::python::list imageShape;
            int imageShapeLen = 3;
            for (int i = 0; i < imageShapeLen; ++i) {
                this->imageShape[i] = cppFormat.imageShape[i];
                imageShape.append(cppFormat.imageShape[i]);
            }
            this->stateFormat = boost::python::make_tuple(tuple(imageShape), cppFormat.varLen);
        }
        return initSuccess;
    }

    float DoomGamePython::makeAction(boost::python::list actionList) {
        // TODO what if isFinished()?
        int listLength = boost::python::len(actionList);
        std::vector<bool> action = std::vector<bool>(listLength);
        for (int i = 0; i < listLength; i++) {
            action[i] = boost::python::extract<bool>(actionList[i]);
        }
        return DoomGame::makeAction(action);
    }

    object DoomGamePython::getState() {
        if (isEpisodeFinished()) {
            return PY_NONE;
        }
        DoomGame::State state = DoomGame::getState();
        PyObject *img = PyArray_SimpleNewFromData(3, this->imageShape, NPY_UBYTE, state.imageBuffer);
        boost::python::handle<> handle(img);
        boost::python::numeric::array npyImg(handle);
        //TODO copy or not?
        if (state.vars.size() > 0) {

            npy_intp varLen = boost::python::extract<int>(this->stateFormat[1]);
            PyObject *vars = PyArray_SimpleNewFromData(1, &varLen, NPY_INT32, state.vars.data());
            boost::python::handle<> handle(vars);
            boost::python::numeric::array npyVars(handle);

            //TODO copy or not?
            return boost::python::make_tuple(state.number, npyImg.copy(), npyVars.copy());
        }
        else {
            //TODO copy or not?
            return boost::python::make_tuple(state.number, npyImg.copy());
        }

    }

    tuple DoomGamePython::getStateFormat() {
        return this->stateFormat;
    }

/* not sure if we need this */
    object DoomGamePython::getLastAction() {
        //TODO
        return PY_NONE;
    }
}