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

    #define PY_NONE bpya::object()

    #if PY_MAJOR_VERSION >= 3
    int
    #else
    void
    #endif
    init_numpy()
    {
        bpyn::array::set_module_and_type("numpy", "ndarray");
        import_array();
    }


    DoomGamePython::DoomGamePython() {
        init_numpy();
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

    std::vector<int> DoomGamePython::pyListToIntVector(bpy::list const &action)
    {
        int listLength = bpy::len(action);
        std::vector<int> properAction = std::vector<int>(listLength);
        for (int i = 0; i < listLength; i++) {
            properAction[i] = bpy::extract<int>(action[i]);
        }
        return properAction;
    }

    void DoomGamePython::setAction(bpy::list const &action) {
        DoomGame::setAction(DoomGamePython::pyListToIntVector(action));
    }

    double DoomGamePython::makeAction(bpy::list const &action) {
        return DoomGame::makeAction(DoomGamePython::pyListToIntVector(action));
    }

    double DoomGamePython::makeAction(bpy::list const &action, unsigned int tics) {
        return DoomGame::makeAction(DoomGamePython::pyListToIntVector(action), tics);
    }

    GameStatePython DoomGamePython::getState() {
        if (this->isEpisodeFinished()) {
            return GameStatePython(this->state.number);
        }

        PyObject *img = PyArray_SimpleNewFromData(3, imageShape, NPY_UBYTE, DoomGame::getGameScreen());
        bpy::handle<> numpyImageHandle = bpy::handle<>(img);
        bpyn::array numpyImage = bpyn::array(numpyImageHandle);

        if (this->state.gameVariables.size() > 0) {
            npy_intp varLen = this->state.gameVariables.size();
            PyObject *vars = PyArray_SimpleNewFromData(1, &varLen, NPY_INT32, this->state.gameVariables.data());
            bpy::handle<> numpyVarsHandle = bpy::handle<>(vars);
            bpyn::array numpyVars = bpyn::array(numpyVarsHandle);

            return GameStatePython(state.number, numpyImage.copy(), numpyVars.copy());
        }
        else {
            return GameStatePython(state.number, numpyImage.copy());
        }

    }

    bpy::list DoomGamePython::getLastAction() {
        bpy::list action;
        for (std::vector<int>::iterator it = DoomGame::lastAction.begin(); it!=DoomGame::lastAction.end(); ++it) {
            action.append(*it);
        }
        return action;
    }

    bpya::object DoomGamePython::getGameScreen(){
        PyObject *img = PyArray_SimpleNewFromData(3, imageShape, NPY_UBYTE, DoomGame::getGameScreen());
        bpy::handle<> numpyImageHandle = bpy::handle<>(img);
        bpyn::array numpyImage = bpyn::array(numpyImageHandle);
        return numpyImage.copy();
    }


    // These functions are workaround for
    // "TypeError: No registered converter was able to produce a C++ rvalue of type std::string from this Python object of type str"
    //  on GCC versions lower then 5
    bool DoomGamePython::loadConfig(bpy::str const &pyPath){
        const char* cPath = bpy::extract<const char *>(pyPath);
        std::string path(cPath);
        return DoomGame::loadConfig(path);
    }

    void DoomGamePython::newEpisode(){
        DoomGame::newEpisode();
    }

    void DoomGamePython::newEpisode(bpy::str const &pyPath){
        const char* cPath = bpy::extract<const char *>(pyPath);
        std::string path(cPath);
        DoomGame::newEpisode(path);
    }

    void DoomGamePython::replayEpisode(bpy::str const &pyPath){
        const char* cPath = bpy::extract<const char *>(pyPath);
        std::string path(cPath);
        DoomGame::replayEpisode(path);
    }

    void DoomGamePython::setViZDoomPath(bpy::str const &pyPath){
        const char* cPath = bpy::extract<const char *>(pyPath);
        std::string path(cPath);
        DoomGame::setViZDoomPath(path);
    }

    void DoomGamePython::setDoomGamePath(bpy::str const &pyPath){
        const char* cPath = bpy::extract<const char *>(pyPath);
        std::string path(cPath);
        DoomGame::setDoomGamePath(path);
    }

    void DoomGamePython::setDoomScenarioPath(bpy::str const &pyPath){
        const char* cPath = bpy::extract<const char *>(pyPath);
        std::string path(cPath);
        DoomGame::setDoomScenarioPath(path);
    }

    void DoomGamePython::setDoomMap(bpy::str const &pyMap){
        const char* cMap = bpy::extract<const char *>(pyMap);
        std::string map(cMap);
        DoomGame::setDoomMap(map);
    }

    void DoomGamePython::setDoomConfigPath(bpy::str const &pyPath){
        const char* cPath = bpy::extract<const char *>(pyPath);
        std::string path(cPath);
        DoomGame::setDoomConfigPath(path);
    }

    void DoomGamePython::addGameArgs(bpy::str const &pyArgs){
        const char* cArgs = bpy::extract<const char *>(pyArgs);
        std::string args(cArgs);
        DoomGame::addGameArgs(args);
    }

    void DoomGamePython::sendGameCommand(bpy::str const &pyCmd){
        const char* cCmd = bpy::extract<const char *>(pyCmd);
        std::string cmd(cCmd);
        DoomGame::sendGameCommand(cmd);
    }

}
