/*
 Copyright (C) 2016 by Wojciech Jaśkowski, Michał Kempka, Grzegorz Runc, Jakub Toczek, Marek Wydmuch

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
*/

#include "ViZDoomGamePython.h"

namespace vizdoom {

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
            
            switch(this->getScreenFormat()) {
                case CRCGCB:
                case CRCGCBDB:
                case CBCGCR:
                case CBCGCRDB:
                case GRAY8:
                case DEPTH_BUFFER8:
                case DOOM_256_COLORS8:
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

    double DoomGamePython::makeAction(boost::python::list const &action) {
        return DoomGame::makeAction(DoomGamePython::pyListToIntVector(action));
    }

    double DoomGamePython::makeAction(boost::python::list const &action, unsigned int tics) {
        return DoomGame::makeAction(DoomGamePython::pyListToIntVector(action), tics);   
    }

    GameStatePython DoomGamePython::getState() {
        if (isEpisodeFinished()) {
            return GameStatePython(this->state.number);
        }

        PyObject *img = PyArray_SimpleNewFromData(3, imageShape, NPY_UBYTE, DoomGame::getGameScreen());
        boost::python::handle<> numpyImageHandle = boost::python::handle<>(img);
        boost::python::numeric::array numpyImage = array(numpyImageHandle);

        if (this->state.gameVariables.size() > 0) {
            npy_intp varLen = this->state.gameVariables.size();
            PyObject *vars = PyArray_SimpleNewFromData(1, &varLen, NPY_INT32, this->state.gameVariables.data());
            boost::python::handle<> numpyVarsHandle = boost::python::handle<>(vars);
            boost::python::numeric::array numpyVars = array(numpyVarsHandle);

            return GameStatePython(state.number, numpyImage.copy(), numpyVars.copy());
        }
        else {
            return GameStatePython(state.number, numpyImage.copy());
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
        PyObject *img = PyArray_SimpleNewFromData(3, imageShape, NPY_UBYTE, DoomGame::getGameScreen());
        boost::python::handle<> numpyImageHandle = boost::python::handle<>(img);
        boost::python::numeric::array numpyImage = array(numpyImageHandle);
        return numpyImage.copy();
    }


    // These functions are workaround for
    // "TypeError: No registered converter was able to produce a C++ rvalue of type std::string from this Python object of type str"
    // on GCC < 5
    bool DoomGamePython::loadConfig(boost::python::str const &pyPath){
        const char* cPath = boost::python::extract<const char *>(pyPath);
        std::string path(cPath);
        return DoomGame::loadConfig(path);
    }

    void DoomGamePython::setViZDoomPath(boost::python::str const &pyPath){
        const char* cPath = boost::python::extract<const char *>(pyPath);
        std::string path(cPath);
        DoomGame::setViZDoomPath(path);
    }

    void DoomGamePython::setDoomGamePath(boost::python::str const &pyPath){
        const char* cPath = boost::python::extract<const char *>(pyPath);
        std::string path(cPath);
        DoomGame::setDoomGamePath(path);
    }

    void DoomGamePython::setDoomScenarioPath(boost::python::str const &pyPath){
        const char* cPath = boost::python::extract<const char *>(pyPath);
        std::string path(cPath);
        DoomGame::setDoomScenarioPath(path);
    }

    void DoomGamePython::setDoomMap(boost::python::str const &pyMap){
        const char* cMap = boost::python::extract<const char *>(pyMap);
        std::string map(cMap);
        DoomGame::setDoomMap(map);
    }

    void DoomGamePython::setDoomConfigPath(boost::python::str const &pyPath){
        const char* cPath = boost::python::extract<const char *>(pyPath);
        std::string path(cPath);
        DoomGame::setDoomConfigPath(path);
    }

    void DoomGamePython::addGameArgs(boost::python::str const &pyArgs){
        const char* cArgs = boost::python::extract<const char *>(pyArgs);
        std::string args(cArgs);
        DoomGame::addGameArgs(args);
    }

    void DoomGamePython::sendGameCommand(boost::python::str const &pyCmd){
        const char* cCmd = boost::python::extract<const char *>(pyCmd);
        std::string cmd(cCmd);
        DoomGame::sendGameCommand(cmd);
    }

}
