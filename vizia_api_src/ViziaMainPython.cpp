#include "ViziaMainPython.h"

using boost::python::tuple;
using boost::python::api::object;

#define PY_NONE object()

ViziaMainPython::ViziaMainPython()
{
    import_array();
}
void ViziaMainPython::init()
{
    bool initSuccess = (ViziaMain::init() == 0);
    /* fill state format */
    if (initSuccess)
    {
        /* Numpy arrays won't work unless this strnage function is envoked.*/
        boost::python::numeric::array::set_module_and_type("numpy", "ndarray");

        ViziaMain::StateFormat cppFormat = ViziaMain::getStateFormat();
        boost::python::list imageShape;
        int imageShapeLen = 3;
        for (int i = 0; i <imageShapeLen; ++i) 
        {
            this->imageShape[i] = cppFormat.imageShape[i];
            imageShape.append( cppFormat.imageShape[i] );
        }
        this->stateFormat = boost::python::make_tuple( tuple(imageShape), cppFormat.varLen );
    }
}

float ViziaMainPython::makeAction(boost::python::list actionList)
{
    // TODO what if isFinished()?
    int listLength = boost::python::len(actionList);
    std::vector<bool> action = std::vector<bool>(listLength);
    for (int i=0; i<listLength; i++)
    {
        action[i]=boost::python::extract<bool>(actionList[i]);
    }
    return ViziaMain::makeAction(action);
}

object ViziaMainPython::getState()
{
    if (isEpisodeFinished())
    {
        return PY_NONE;
    }
    ViziaMain::State state = ViziaMain::getState();
    //TODO convert the image state to numpy array
    PyObject* img = PyArray_SimpleNewFromData(3, this->imageShape, NPY_UBYTE, state.imageBuffer);
    boost::python::handle<> handle( img );
    boost::python::numeric::array npyImg( handle );

    //TODO copy or no?
    if (state.vars != NULL)
    {

        npy_intp varLen = boost::python::extract<int>(this->stateFormat[1]);
        PyObject* vars = PyArray_SimpleNewFromData(1, &varLen, NPY_INT32, state.vars);
        boost::python::handle<> handle( vars );
        boost::python::numeric::array npyVars( handle );

        //TODO copy or no?
        return boost::python::make_tuple(state.number, npyImg.copy(), npyVars.copy());
    }
    else
    {
        //TODO copy or no?
        return boost::python::make_tuple(state.number, npyImg.copy());
    }
    
}

tuple ViziaMainPython::getStateFormat()
{
    return this->stateFormat;
}
       
/* not sure if we need this */
object ViziaMainPython::getLastAction()
{
    //TODO
    return PY_NONE;
}