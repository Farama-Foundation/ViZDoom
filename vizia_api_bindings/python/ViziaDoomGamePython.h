#ifndef __VIZIA_MAIN_PYTHON_H__
#define __VIZIA_MAIN_PYTHON_H__

#define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION

#include "ViziaDoomGame.h"
#include <iostream>
#include <vector>
#include <Python.h>

#include <boost/python.hpp>
#include <boost/python/list.hpp>
#include <boost/python/object.hpp>
#include <boost/python/tuple.hpp>
#include <numpy/ndarrayobject.h>
#include <numpy/npy_math.h>

namespace Vizia {
    using boost::python::api::object;
/* C++ code to expose C arrays as python objects */
    class DoomGamePython : public DoomGame {
        
    public:
        struct PythonState
        {
            int number;
            object imageBuffer;
            object vars;
            PythonState(int n, object buf, object v ):number(n),imageBuffer(buf),vars(v){}
            PythonState(int n, object buf):number(n),imageBuffer(buf){}
            PythonState(int n):number(n){}
        };
        DoomGamePython();
        bool init();
        void setAction(boost::python::list &action);
        PythonState getState();
        boost::python::list getLastAction();
        object getGameScreen();
        float makeAction(boost::python::list &action);
        float makeAction(boost::python::list &action, unsigned int tics);

    private:

        npy_intp imageShape[3];

    };


}

#endif
