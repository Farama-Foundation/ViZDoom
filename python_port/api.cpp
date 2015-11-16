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
static PyObject *api_average_best_result(PyObject *self, PyObject *args);

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
    {"average_best_result", api_average_best_result, METH_VARARGS, NULL},


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
    /* Initialization */
	int _x;
    int _y;
    //optional arguments
    int _random_background;
    int _max_moves;
    double _living_reward;
    double _miss_penalty;
    double _hit_reward;
    int _ammo; 
	double float_ammo;

    /* Parse arguments */
    if (!PyArg_ParseTuple(args, "iiiidddd", &_x, &_y, &_random_background, &_max_moves,
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
    /* Create state format and action format objects */
    PyObject* image_state_format_tuple = PyTuple_New(get_state_format()->image_shape_len);

    for(int i = 0; i < get_state_format()->image_shape_len; i++)
    {
        PyTuple_SetItem(image_state_format_tuple,i,PyInt_FromLong(get_state_format()->image_shape[i]));
    }
    state_format = PyTuple_Pack(2, image_state_format_tuple, PyInt_FromLong(get_state_format()->misc_len));    

    /* Create state object */
    npy_intp* img_shape = new npy_intp[get_state_format()-> image_shape_len];
    for (int i = 0; i<get_state_format()->image_shape_len; ++i)
    {
        img_shape[i] = get_state_format()->image_shape[i];
    }
    PyObject* img_state = PyArray_SimpleNewFromData(get_state_format()->image_shape_len, img_shape, NPY_FLOAT32, get_state()->image);
    delete[] img_shape;
    
    int misc_len = get_state_format()->misc_len;
    if( misc_len > 0)
    {
        npy_intp misc_dimensions[] = { 1 };
        misc_dimensions[0] = misc_len;
        PyObject* misc_state = PyArray_SimpleNewFromData(misc_len, misc_dimensions, NPY_FLOAT32, get_state()->misc);
        state = PyTuple_Pack(2,img_state, misc_state);
    }
    else
    {
        state = PyTuple_Pack(1, img_state);
    }

    Py_RETURN_NONE;
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
   	double summary_reward = get_summary_reward();	
	PyObject *ret = Py_BuildValue("d", summary_reward);
    Py_XINCREF(ret);
	return ret;
}

static PyObject *api_new_episode(PyObject *self, PyObject *args)
{
    new_episode();
	Py_RETURN_NONE;
}

static PyObject *api_make_action(PyObject *self, PyObject *args)
{
    if (is_finished())
    {
        // TODO throw some exception
        return NULL;
    }
	PyObject *pyobj_action;
	if (!PyArg_ParseTuple(args, "O", &pyobj_action))
	{
        // TODO throw some exception
		return NULL;
	}
    pyobj_action = PyArray_FROM_OTF(pyobj_action, NPY_INT, NPY_IN_ARRAY);
 	if (pyobj_action == NULL) 
    {
       	Py_XDECREF(pyobj_action);
        // TODO throw some exception
       	return NULL;
    }

	int *action = (int*)PyArray_DATA(pyobj_action);
	double reward = make_action(action);

    Py_XDECREF(pyobj_action);

    PyObject *pyobj_reward = Py_BuildValue("d", reward);
    Py_XINCREF(pyobj_reward);
	return pyobj_reward;

}

static PyObject *api_get_state(PyObject *self, PyObject *args)
{
    if( is_finished() )
    {
        Py_RETURN_NONE;
    }
    Py_XINCREF(state);
    return state;
}

static PyObject *api_average_best_result(PyObject *self, PyObject *args)
{
    PyObject *pyobj_reward = Py_BuildValue("d", average_best_result());
    Py_XINCREF(pyobj_reward);
    return pyobj_reward;
}
