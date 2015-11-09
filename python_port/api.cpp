#include <Python.h>
#include <numpy/arrayobject.h>
#include "chi2.h"
#include "dot_shooting.h"
/* Docstrings */
static char module_docstring[] =
    "Vizia api in python";


/* Available functions */
static int xstat,ystat;
static PyObject *api_chi2(PyObject *self, PyObject *args);
static PyObject *api_nothing(PyObject *self, PyObject *args);
static PyObject *api_dictionary(PyObject *self, PyObject *args);
static PyObject *api_ndarray(PyObject *self, PyObject *args);

static PyObject *api_a(PyObject *self, PyObject *args);//api_get_misc_state(PyObject *self, PyObject *args);

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
    {"chi2", api_chi2, METH_VARARGS, NULL},
    {"dictionary", api_dictionary, METH_VARARGS, NULL},
    {"nothing", api_nothing, METH_VARARGS, NULL},
    {"ndarray", api_ndarray, METH_VARARGS, NULL},
    
    {"init", api_init, METH_VARARGS, NULL},
    {"is_finished", api_is_finished, METH_VARARGS, NULL},
    {"get_state_format", api_get_state_format, METH_VARARGS, NULL},
    {"get_action_format", api_get_action_format, METH_VARARGS, NULL},
    {"get_summary_reward", api_get_summary_reward, METH_VARARGS, NULL},
	{ "a",api_a,METH_VARARGS, NULL},
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
/* game functions */


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
    int _ammo; //may be a number or np.inf
	
    if (!PyArg_ParseTuple(args, "iiiiffff", &_x, &_y, &_random_background, &_max_moves,
                                         &_living_reward,&_miss_penalty,&_hit_reward,&_ammo))
    {
        return NULL;
    }   
    init(_x,_y, _random_background, _max_moves, _living_reward,_miss_penalty,_hit_reward,int(_ammo));

    return Py_None;
}

static PyObject *api_is_finished(PyObject *self, PyObject *args)
{
	int value=is_finished();	
	PyObject *ret = Py_BuildValue("i", value);
	return ret;
}

static PyObject *api_get_state_format(PyObject *self, PyObject *args)
{
	int* value=get_state_format();

    npy_intp* dims = (npy_intp*)malloc(sizeof(npy_intp));
    int n =4;
    dims[0]=4;

    int* data = (int*)malloc(n *sizeof(int));
    for (int i =0;i<n;i++)
    {
        data[i] = value[i];
    }

    PyObject* array = PyArray_SimpleNewFromData(1, dims, NPY_INT32, data);
    PyObject* list = PyList_New(0);
    PyList_Append(list,array);
    return list; 

}
static PyObject *api_a(PyObject *self, PyObject *args)
{
	float* value=get_misc_state();
	int b=1;
	PyObject *result = PyList_New(b); 
  for (int i = 0; i < b; ++i) {
    PyList_SetItem(result, i, PyFloat_FromDouble(value[i]));
  }

	return result;
}

static PyObject *api_get_action_format(PyObject *self, PyObject *args)
{
	int value=get_action_format();	
	PyObject *ret = Py_BuildValue("i", value);
	return ret;

}
static PyObject *api_get_summary_reward(PyObject *self, PyObject *args)
{
   	int value=get_summary_reward();	
	PyObject *ret = Py_BuildValue("i", value);
	return ret;
}
static PyObject *api_new_episode(PyObject *self, PyObject *args)
{
    	new_episode();
	return Py_None;
}
static PyObject *api_make_action(PyObject *self, PyObject *args)
{
	PyObject *x_obj;
	if (!PyArg_ParseTuple(args, "O", &x_obj))
	{
		return NULL;
	}

    /* Interpret the input objects as numpy arrays. */
    PyObject *x_array = PyArray_FROM_OTF(x_obj, NPY_INT, NPY_IN_ARRAY);
 	if (x_array == NULL) {
        	Py_XDECREF(x_array);
        	return NULL;
    	}
	int *x    = (int*)PyArray_DATA(x_array);
	float value=make_action(x);
    	PyObject *ret = Py_BuildValue("f", value);
	return ret;
}
static PyObject *api_get_state(PyObject *self, PyObject *args)
{

	npy_intp* dims = (npy_intp*)malloc(sizeof(npy_intp));
    int n =xstat*ystat;
    	dims[0]=xstat;
	dims[1]=ystat;

	float** _stat=get_image_state();

    float* data = (float*)malloc(n *sizeof(float));
    for (int i =0;i<xstat;i++)
    {
        for (int j =0;j<ystat;j++)
    {
        data[(i*ystat+j)] = _stat[i][j];
    }
    }
    //return Py_None;
    PyObject* array = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, data);
    PyObject* list = PyList_New(0);
    PyList_Append(list,array);
	dims[0]=1;
	float* temp;
	temp=get_misc_state();
	PyObject* array2 = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, temp);
    PyList_Append(list,array2);
    return list;
    //return Py_None;
}
//////////////////////
static PyObject *api_dictionary(PyObject *self, PyObject *args)
{
    PyObject* dict = PyDict_New();
    PyObject* key = PyString_FromString("name");
    PyObject* value = PyString_FromString("whatever");
    PyDict_SetItem (dict,key,value);
    PyObject* list = PyList_New(0);
    PyList_Append(list,dict);
    return list;
}
static PyObject *api_ndarray(PyObject *self, PyObject *args)
{
    npy_intp* dims = (npy_intp*)malloc(sizeof(npy_intp));
    int n =10;
	printf("%d\n",xstat);
    dims[0]=3;

    float* data = (float*)malloc(n *sizeof(float));
    for (int i =0;i<n;i++)
    {
        data[i] = 9;
    }
    //return Py_None;
    PyObject* array = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, data);
    PyObject* array2 = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, data);
    PyObject* list = PyList_New(0);
    PyList_Append(list,array);
    PyList_Append(list,array2);
    return list;
}
static PyObject *api_nothing(PyObject *self, PyObject *args)
{
    return Py_None;
}
static PyObject *api_chi2(PyObject *self, PyObject *args)
{
    double m, b;
    PyObject *x_obj, *y_obj, *yerr_obj;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "ddOOO", &m, &b, &x_obj, &y_obj,
                                         &yerr_obj))
        return NULL;

    /* Interpret the input objects as numpy arrays. */
    PyObject *x_array = PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *y_array = PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *yerr_array = PyArray_FROM_OTF(yerr_obj, NPY_DOUBLE,
                                            NPY_IN_ARRAY);

    /* If that didn't work, throw an exception. */
    if (x_array == NULL || y_array == NULL || yerr_array == NULL) {
        Py_XDECREF(x_array);
        Py_XDECREF(y_array);
        Py_XDECREF(yerr_array);
        return NULL;
    }

    /* How many data points are there? */
    int N = (int)PyArray_DIM(x_array, 0);

    /* Get pointers to the data as C-types. */
    double *x    = (double*)PyArray_DATA(x_array);
    double *y    = (double*)PyArray_DATA(y_array);
    double *yerr = (double*)PyArray_DATA(yerr_array);

    /* Call the external C function to compute the chi-squared. */
    double value = chi2(m, b, x, y, yerr, N);

    /* Clean up. */
    Py_DECREF(x_array);
    Py_DECREF(y_array);
    Py_DECREF(yerr_array);

    if (value < 0.0) {
        PyErr_SetString(PyExc_RuntimeError,
                    "Chi-squared returned an impossible value.");
        return NULL;
    }

    /* Build the output tuple */
    PyObject *ret = Py_BuildValue("d", value);
    return ret;
}
