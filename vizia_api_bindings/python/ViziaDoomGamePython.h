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

/* C++ code to expose C arrays as python objects */
    class DoomGamePython : public DoomGame {

    public:
        DoomGamePython();
        ~DoomGamePython();
        bool init();
        float makeAction(boost::python::list actionList);
        boost::python::api::object getState();
        boost::python::api::object getLastAction();

    private:

        boost::python::numeric::array* numpyImage;
        boost::python::numeric::array* numpyVars;

    };

}

#endif
