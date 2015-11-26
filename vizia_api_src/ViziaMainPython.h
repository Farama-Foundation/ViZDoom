#ifndef __VIZIA_MAIN_PYTHON_H__
#define __VIZIA_MAIN_PYTHON_H__

#include "ViziaMain.h"
#include <iostream>
#include <vector>
#include <Python.h>

#include <boost/python.hpp>
#include <boost/python/list.hpp>
#include <boost/python/object.hpp>
#include <boost/python/tuple.hpp>
#include <numpy/ndarrayobject.h>

#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>

/* C++ code to expose C arrays as python objects */
class ViziaMainPython: public ViziaMain 
{
	
    public:
        ViziaMainPython();
        void init();
    	float makeAction(boost::python::list actionList);
        boost::python::api::object getState();
        boost::python::tuple getStateFormat();
        boost::python::api::object getLastAction();

    private:

        boost::python::tuple stateFormat;
        npy_intp imageShape[3];

};


#endif