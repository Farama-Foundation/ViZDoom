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

#include <cstddef>
#include <cstring>

namespace vizdoom {

    #if PY_MAJOR_VERSION >= 3
        void* init_numpy() {
            import_array();
            return NULL;
        }
    #else
        void init_numpy() {
            import_array();
        }
    #endif

    DoomGamePython::DoomGamePython() {
        init_numpy();
    }

    void DoomGamePython::setAction(pyb::list const &pyAction) {
        auto action = DoomGamePython::pyListToVector<double>(pyAction);
        ReleaseGIL gil = ReleaseGIL();
        DoomGame::setAction(action);
    }

    double DoomGamePython::makeAction(pyb::list const &pyAction, unsigned int tics) {
        auto action = DoomGamePython::pyListToVector<double>(pyAction);
        ReleaseGIL gil = ReleaseGIL();
        return DoomGame::makeAction(action, tics);
    }

    GameStatePython* DoomGamePython::getState() {
        if (this->state == nullptr) return nullptr;

        // TODO: the following line causes:
        // Fatal Python error: PyEval_SaveThread: NULL tstate
        //ReleaseGIL gil = ReleaseGIL();
        this->pyState = new GameStatePython();

        this->pyState->number = this->state->number;
        this->pyState->tic = this->state->tic;

        /* Update buffers */
        this->updateBuffersShapes();
        int colorDims = 3;
        if (this->getScreenChannels() == 1) colorDims = 2;

        if (this->state->screenBuffer != nullptr)
            this->pyState->screenBuffer = this->dataToNumpyArray(colorDims, this->colorShape, NPY_UBYTE, this->state->screenBuffer->data());
        else this->pyState->screenBuffer = pyb::none();

        if (this->state->depthBuffer != nullptr)
            this->pyState->depthBuffer = this->dataToNumpyArray(2, this->grayShape, NPY_UBYTE, this->state->depthBuffer->data());
        else this->pyState->depthBuffer = pyb::none();

        if (this->state->labelsBuffer != nullptr) {
            this->pyState->labelsBuffer = this->dataToNumpyArray(2, this->grayShape, NPY_UBYTE, this->state->labelsBuffer->data());

            /* Update labels */
            this->pyState->labels = DoomGamePython::vectorToPyList<Label>(this->state->labels);
        }  else {
            this->pyState->labelsBuffer = pyb::none();
            this->pyState->labels = pyb::list();
        }

        if (this->state->automapBuffer != nullptr)
            this->pyState->automapBuffer = this->dataToNumpyArray(colorDims, this->colorShape, NPY_UBYTE, this->state->automapBuffer->data());
        else this->pyState->automapBuffer = pyb::none();

        /* Updates vars */
        if (this->state->gameVariables.size() > 0) {
            // Numpy array version
            npy_intp shape = this->state->gameVariables.size();
            this->pyState->gameVariables = dataToNumpyArray(1, &shape, NPY_DOUBLE, this->state->gameVariables.data());

            // Python list version
            //this->pyState->gameVariables = DoomGamePython::vectorToPyList<double>(this->state->gameVariables);
        }
        else this->pyState->gameVariables = pyb::none();

        /* Update objects */
        if (this->isObjectsInfoEnabled()) {
            this->pyState->objects = DoomGamePython::vectorToPyList<Object>(this->state->objects);
        } else this->pyState->objects = pyb::list();

        /* Update sectors */
        if (this->isSectorsInfoEnabled()) {
            pyb::list pySectors;
            for (auto& sector : this->state->sectors){
                SectorPython pySector;
                pySector.floorHeight = sector.floorHeight;
                pySector.ceilingHeight = sector.ceilingHeight;
                pySector.lines = DoomGamePython::vectorToPyList<Line>(sector.lines);
                pySectors.append(pySector);
            }
            this->pyState->sectors = pySectors;
            //this->pyState->sectors = DoomGamePython::vectorToPyList<Sectors>(this->state->objects);
        } else this->pyState->sectors = pyb::list();

        return this->pyState;
    }

    ServerStatePython* DoomGamePython::getServerState() {
        ServerStatePython* pyServerState = new ServerStatePython();

        pyServerState->tic = this->doomController->getMapTic();
        pyServerState->playerCount = this->doomController->getPlayerCount();

        pyb::list pyPlayersInGame, pyPlayersNames, pyPlayersFrags,
                pyPlayersAfk, pyPlayersLastActionTic, pyPlayersLastKillTic;
        for(int i = 0; i < MAX_PLAYERS; ++i) {
            pyPlayersInGame.append(this->doomController->isPlayerInGame(i));
            pyPlayersNames.append(pyb::str(this->doomController->getPlayerName(i).c_str()));
            pyPlayersFrags.append(this->doomController->getPlayerFrags(i));
            pyPlayersAfk.append(this->doomController->isPlayerAfk(i));
            pyPlayersLastActionTic.append(this->doomController->getPlayerLastActionTic(i));
            pyPlayersLastKillTic.append(this->doomController->getPlayerLastKillTic(i));
        }

        pyServerState->playersInGame = pyPlayersInGame;
        pyServerState->playersNames = pyPlayersNames;
        pyServerState->playersFrags = pyPlayersFrags;
        pyServerState->playersAfk = pyPlayersAfk;
        pyServerState->playersLastActionTic = pyPlayersLastActionTic;
        pyServerState->playersLastKillTic = pyPlayersLastKillTic;

        return pyServerState;
    }

    pyb::list DoomGamePython::getLastAction() {
        return DoomGamePython::vectorToPyList(this->lastAction);
    }

    pyb::list DoomGamePython::getAvailableButtons(){
        return DoomGamePython::vectorToPyList(this->availableButtons);
    }

    void DoomGamePython::setAvailableButtons(pyb::list const &pyButtons){
        DoomGame::setAvailableButtons(DoomGamePython::pyListToVector<Button>(pyButtons));
    }

    pyb::list DoomGamePython::getAvailableGameVariables(){
        return DoomGamePython::vectorToPyList(this->availableGameVariables);
    }

    void DoomGamePython::setAvailableGameVariables(pyb::list const &pyGameVariables){
        DoomGame::setAvailableGameVariables(DoomGamePython::pyListToVector<GameVariable>(pyGameVariables));
    }

    // These functions are wrapped for manual GIL management
    void DoomGamePython::init(){
        ReleaseGIL gil = ReleaseGIL();
        DoomGame::init();
    }

    void DoomGamePython::advanceAction(unsigned int tics, bool updateState){
        ReleaseGIL gil = ReleaseGIL();
        DoomGame::advanceAction(tics, updateState);
    }

    void DoomGamePython::respawnPlayer(){
        ReleaseGIL gil = ReleaseGIL();
        DoomGame::respawnPlayer();
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


    template<class T> pyb::list DoomGamePython::vectorToPyList(const std::vector<T>& vector){
        pyb::list pyList;
        for (auto& i : vector) pyList.append(i);
        return pyList;
    }

    template<class T> std::vector<T> DoomGamePython::pyListToVector(pyb::list const &pyList){
        size_t pyListLength = pyb::len(pyList);
        std::vector<T> vector = std::vector<T>(pyListLength);
        for (size_t i = 0; i < pyListLength; ++i) vector[i] = pyb::cast<T>(pyList[i]);
        return vector;
    }

    pyb::object DoomGamePython::dataToNumpyArray(int dims, npy_intp *shape, int type, void *data) {
        PyObject *pyArray = PyArray_SimpleNewFromData(dims, shape, type, data);
        /* This line makes a copy: */
        PyObject *pyArrayCopied = PyArray_FROM_OTF(pyArray, type, NPY_ARRAY_ENSURECOPY | NPY_ARRAY_ENSUREARRAY);
        /* And this line gets rid of the old object which caused a memory leak: */
        Py_DECREF(pyArray);

        pyb::handle numpyArrayHandle = pyb::handle(pyArrayCopied);
        pyb::object numpyArray = pyb::reinterpret_steal<pyb::object>(numpyArrayHandle);

        return numpyArray;
    }
}
