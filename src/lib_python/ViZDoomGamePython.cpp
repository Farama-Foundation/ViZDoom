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
#include "ViZDoomController.h"

namespace vizdoom {

    #define PY_NONE bpya::object()

    #if PY_MAJOR_VERSION >= 3
    int
    #else
    void
    #endif
    init_numpy() {
        bpyn::array::set_module_and_type("numpy", "ndarray");
        import_array();
    }

    DoomGamePython::DoomGamePython() {
        init_numpy();
    }

    void DoomGamePython::setAction(bpy::list const &pyAction) {
        DoomGame::setAction(DoomGamePython::pyListToVector<int>(pyAction));
    }

    double DoomGamePython::makeAction(bpy::list const &pyAction, unsigned int tics) {
        return DoomGame::makeAction(DoomGamePython::pyListToVector<int>(pyAction), tics);
    }

    GameStatePython DoomGamePython::getState() {

        GameStatePython pyState;

        if (this->state == nullptr) return pyState;

        pyState.number = this->state->number;

        if (this->isEpisodeFinished()) return pyState;

        this->updateBuffersShapes();
        int colorDims = 3;
        if (this->getScreenChannels() == 1) colorDims = 2;

        if (this->state->screenBuffer != nullptr)
            pyState.screenBuffer = this->dataToNumpyArray(colorDims, this->colorShape, NPY_UBYTE, this->state->screenBuffer->data());
        if (this->state->depthBuffer != nullptr)
            pyState.depthBuffer = this->dataToNumpyArray(2, this->grayShape, NPY_UBYTE, this->state->depthBuffer->data());
        if (this->state->labelsBuffer != nullptr)
            pyState.labelsBuffer = this->dataToNumpyArray(2, this->grayShape, NPY_UBYTE, this->state->labelsBuffer->data());
        if (this->state->automapBuffer != nullptr)
            pyState.automapBuffer = this->dataToNumpyArray(colorDims, this->colorShape, NPY_UBYTE, this->state->automapBuffer->data());

        if (this->state->gameVariables.size() > 0) {
            // Numpy array version
            npy_intp shape = this->state->gameVariables.size();
            pyState.gameVariables = dataToNumpyArray(1, &shape, NPY_DOUBLE, this->state->gameVariables.data());

            // Python list version
            //pyState.gameVariables = DoomGamePython::vectorToPyList<int>(this->state->gameVariables);
        }

        if(this->state->labels.size() > 0){
            bpy::list pyLabels;
            for(auto i = this->state->labels.begin(); i != this->state->labels.end(); ++i){
                LabelPython pyLabel;
                pyLabel.objectId = i->objectId;
                pyLabel.objectName = bpy::str(i->objectName.c_str());
                pyLabel.value = i->value;
                pyLabel.objectPositionX = i->objectPositionX;
                pyLabel.objectPositionY = i->objectPositionY;
                pyLabel.objectPositionZ = i->objectPositionZ;
                pyLabels.append(pyLabel);
            }

            pyState.labels = pyLabels;
        }

        return pyState;
    }

    bpy::list DoomGamePython::getLastAction() {
        return DoomGamePython::vectorToPyList(this->lastAction);
    }

    bpy::list DoomGamePython::getAvailableButtons(){
        return DoomGamePython::vectorToPyList(this->availableButtons);
    }

    void DoomGamePython::setAvailableButtons(bpy::list const &pyButtons){
        DoomGame::setAvailableButtons(DoomGamePython::pyListToVector<Button>(pyButtons));
    }

    bpy::list DoomGamePython::getAvailableGameVariables(){
        return DoomGamePython::vectorToPyList(this->availableGameVariables);
    }

    void DoomGamePython::setAvailableGameVariables(bpy::list const &pyGameVariables){
        DoomGame::setAvailableGameVariables(DoomGamePython::pyListToVector<GameVariable>(pyGameVariables));
    }


    // These functions are workaround for
    // "TypeError: No registered converter was able to produce a C++ rvalue of type std::string from this Python object of type str"
    // on GCC versions lower then 5
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

    void DoomGamePython::replayEpisode(bpy::str const &pyPath, unsigned int player){
        const char* cPath = bpy::extract<const char *>(pyPath);
        std::string path(cPath);
        DoomGame::replayEpisode(path, player);
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


    void DoomGamePython::updateBuffersShapes(){
        int channels = this->getScreenChannels();
        int width = this->getScreenWidth();
        int height = this->getScreenHeight();

        switch(this->getScreenFormat()){
            case CRCGCB:
            case CBCGCR:
                this->colorShape[0] = channels;
                this->colorShape[1] = height;
                this->colorShape[2] = width;
                break;

            default:
                this->colorShape[0] = height;
                this->colorShape[1] = width;
                this->colorShape[2] = channels;
        }

        this->grayShape[0] = height;
        this->grayShape[1] = width;
    }


    template<class T> bpy::list DoomGamePython::vectorToPyList(const std::vector<T>& vector){
        bpy::list pyList;
        for (auto i : vector) pyList.append(i);
        return pyList;
    }

    template<class T> std::vector<T> DoomGamePython::pyListToVector(bpy::list const &pyList){
        size_t pyListLength = bpy::len(pyList);
        std::vector<T> vector = std::vector<T>(pyListLength);
        for (size_t i = 0; i < pyListLength; ++i) vector[i] = bpy::extract<T>(pyList[i]);
        return vector;
    }

    bpy::object DoomGamePython::dataToNumpyArray(int dims, npy_intp * shape, int type, void * data){
        PyObject *pyArray = PyArray_SimpleNewFromData(dims, shape, type, data);
        /* This line makes a copy: */
        pyArray = PyArray_FROM_OTF(pyArray, type, NPY_ARRAY_ENSURECOPY | NPY_ARRAY_ENSUREARRAY);
        bpy::handle<> numpyHandle = bpy::handle<>(pyArray);
        bpy::object numpyArray = bpy::object(numpyHandle);

        /* This line caused occasional segfaults in python3 */
        //bpyn::array numpyArray = bpyn::array(numpyHandle);

        return numpyArray;
    }
}
