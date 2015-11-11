#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>
#include "dot_shooting.h"
#include <cstdio>
/* Docstrings */
static char module_docstring[] =
    "Vizia api in python";


/* Available functions */

static PyObject *api_init(PyObject *self, PyObject *args);
static PyObject *api_is_finished(PyObject *self, PyObject *args);
static PyObject *api_get_state_format(PyObject *self, PyObject *args);
static PyObject *api_get_action_format(PyObject *self, PyObject *args);
static PyObject *api_get_summary_reward(PyObject *self, PyObject *args);
static PyObject *api_new_episode(PyObject *self, PyObject *args);
static PyObject *api_make_action(PyObject *self, PyObject *args);
static PyObject *api_get_state(PyObject *self, PyObject *args);

/* Module specification */
static PyMethodDef module_methods[] = {
    {"init", api_init, METH_VARARGS, NULL},
    {"is_finished", api_is_finished, METH_VARARGS, NULL},
    {"get_state_format", api_get_state_format, METH_VARARGS, NULL},
    {"get_action_format", api_get_action_format, METH_VARARGS, NULL},
    {"get_summary_reward", api_get_summary_reward, METH_VARARGS, NULL},
    {"new_episode", api_new_episode, METH_VARARGS, NULL},
    {"make_action", api_make_action, METH_VARARGS, NULL},
    {"get_state", api_get_state, METH_VARARGS, NULL},



};

/* Initialize the module */
PyMODINIT_FUNC initapi(void)
{
    PyObject *m = Py_InitModule3("api", module_methods, module_docstring);
    if (m == NULL)
        return;

    /* Load `numpy` functionality. */
    import_array();
}


PyObject* state_format;
PyObject* action_format;
PyObject* state;


static PyObject * api_init(PyObject *self, PyObject *args)
{
	int _x;
    int _y;
    //optional arguments
    int _random_background;
    int _max_moves;
    float _living_reward;
    float _miss_penalty;
    float _hit_reward;
    int _ammo; 
	float float_ammo;

    /* Parse arguments */
    if (!PyArg_ParseTuple(args, "iiiiffff", &_x, &_y, &_random_background, &_max_moves,
                                         &_living_reward,&_miss_penalty,&_hit_reward,&float_ammo))
    {
        return NULL;
    }   
    /* Handle ammo infinite value */
    if(NPY_INFINITY == float_ammo)
    {
        _ammo = -1;
    }
    else
    {
        _ammo = (int)float_ammo;
    }
    /* Run the proper code in c++ api */
    init(_x,_y, (int)_random_background, _max_moves, _living_reward,_miss_penalty,_hit_reward,_ammo);
    _x += (_x +1)%2;
    /* Create state format and action format objects */
    PyObject* image_state_tuple = PyTuple_Pack(2,PyInt_FromLong(_y),PyInt_FromLong(_x));
    npy_intp image_dimensions[] = {_y,_x};
    PyObject* img_state = PyArray_SimpleNewFromData(2, image_dimensions, NPY_FLOAT32, get_image_state());
    
    if(_ammo == -1)
    {
        state_format = PyTuple_Pack(1,image_state_tuple);
        state = PyTuple_Pack(1,img_state);
    }
    else
    {
        state_format = PyTuple_Pack(2, image_state_tuple, PyInt_FromLong(_ammo));
        npy_intp misc_dimensions[] = { 1 };
        PyObject* misc_state = PyArray_SimpleNewFromData(1, misc_dimensions, NPY_FLOAT32, get_misc_state());
        state = PyTuple_Pack(2,img_state, misc_state);
    }


    return Py_None;
}

static PyObject *api_is_finished(PyObject *self, PyObject *args)
{
    PyObject * finished = PyBool_FromLong(is_finished());
    Py_XINCREF(finished);
	return finished;
}

static PyObject *api_get_state_format(PyObject *self, PyObject *args)
{
    Py_XINCREF(state_format);
	return state_format;
}

static PyObject *api_get_action_format(PyObject *self, PyObject *args)
{
    PyObject* format = Py_BuildValue("i", get_action_format());
    Py_XINCREF(format);
	return format;
}

static PyObject *api_get_summary_reward(PyObject *self, PyObject *args)
{
   	float summary_reward = get_summary_reward();	
	PyObject *ret = Py_BuildValue("f", summary_reward);
    Py_XINCREF(ret);
	return ret;
}

static PyObject *api_new_episode(PyObject *self, PyObject *args)
{
    new_episode();
	return Py_None;
}

static PyObject *api_make_action(PyObject *self, PyObject *args)
{
    
	PyObject *pyobj_action;
	if (!PyArg_ParseTuple(args, "O", &pyobj_action))
	{
		return NULL;
	}
    pyobj_action = PyArray_FROM_OTF(pyobj_action, NPY_INT, NPY_IN_ARRAY);
 	if (pyobj_action == NULL) 
    {
       	Py_XDECREF(pyobj_action);
       	return NULL;
    }

	int *action = (int*)PyArray_DATA(pyobj_action);
	float reward = make_action(action);

    Py_XDECREF(action);

    PyObject *pyobj_reward = Py_BuildValue("f", reward);
	return pyobj_reward;

}

static PyObject *api_get_state(PyObject *self, PyObject *args)
{
    Py_XINCREF(state);
    return state;
}
