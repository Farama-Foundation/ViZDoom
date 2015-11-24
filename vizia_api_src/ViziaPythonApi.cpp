#include "ViziaMain.h"
#include <iostream>
#include <vector>
#include <Python.h>

#include <boost/python/list.hpp>
#include <boost/python/object.hpp>
#include <boost/python/tuple.hpp>
#include <numpy/ndarrayobject.h>

#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>

using boost::python::tuple;
using boost::python::api::object;

#define PY_NONE object()

/* C++ code to expose C arrays as python objects */
class ViziaPythonApi: public ViziaMain 
{
	
    public:
        ViziaPythonApi()
        {
            import_array();
        }

        ~ViziaPythonApi()
        {
            this->close();
        }

        void init()
        {
            bool init_success = (ViziaMain::init() == 0);
            /* fill state format */
            if (init_success)
            {
                /* Numpy arrays won't work unless this strnage function is envoked.*/
                boost::python::numeric::array::set_module_and_type("numpy", "ndarray");

                ViziaMain::StateFormat cpp_format = ViziaMain::getStateFormat();
                boost::python::list image_shape;
                int image_shape_len = 3;
                for (int i = 0; i <image_shape_len; ++i) 
                {
                    image_shape.append(cpp_format.image_shape[i]);
                }
                this->state_format = boost::python::make_tuple(tuple(image_shape),cpp_format.var_len);
            }
        }

    	float makeAction(boost::python::list action_list)
        {
            // TODO what if isFinished()?
    		int list_length = boost::python::len(action_list);
    		std::vector<bool> action = std::vector<bool>(list_length);
    		for (int i=0; i<list_length; i++)
    		{
    			action[i]=boost::python::extract<bool>(action_list[i]);
    		}
    		return ViziaMain::makeAction(action);
    	}

        object getState()
        {
            if (isEpisodeFinished())
            {
                return PY_NONE;
            }
            ViziaMain::State state = ViziaMain::getState();
            //TODO convert the image state to numpy array
            if (state.vars != NULL)
            {

                npy_intp var_len = boost::python::extract<int>(state_format[1]);
                PyObject* vars = PyArray_SimpleNewFromData(1, &var_len, NPY_INT32, state.vars);
                boost::python::handle<> handle( vars );
                boost::python::numeric::array npy_vars( handle );
                
                return boost::python::make_tuple(state.number, PY_NONE, npy_vars.copy());
            }
            else
            {
                return boost::python::make_tuple(state.number, PY_NONE);
            }
            
        }

        tuple getStateFormat()
        {
           return this->state_format;
        }
       
        /* not sure if we need this */
        object getLastAction()
        {
            //TODO
            return PY_NONE;
        }

    private:

        tuple state_format;

};
