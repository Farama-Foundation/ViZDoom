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
    DoomGamePython::DoomGamePython() {
        this->grayShape.resize(2);
        this->audioShape.resize(2);
        this->variablesShape.resize(1);
    }

//    void DoomGamePython::setAction(pyb::list const &pyAction) {
//        auto action = DoomGamePython::pyListToVector<double>(pyAction);
//        ReleaseGIL gil = ReleaseGIL();
//        DoomGame::setAction(action);
//    }

//    double DoomGamePython::makeAction(pyb::list const &pyAction, unsigned int tics) {
//        auto action = DoomGamePython::pyListToVector<double>(pyAction);
//        ReleaseGIL gil = ReleaseGIL();
//        return DoomGame::makeAction(action, tics);
//    }

    void DoomGamePython::setAction(pyb::object const &pyAction) {
        auto action = DoomGamePython::pyObjectToVector<double>(pyAction);
        ReleaseGIL gil = ReleaseGIL();
        DoomGame::setAction(action);
    }

    double DoomGamePython::makeAction(pyb::object const &pyAction, unsigned int tics) {
        auto action = DoomGamePython::pyObjectToVector<double>(pyAction);
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

        if (this->state->screenBuffer != nullptr)
            this->pyState->screenBuffer = this->dataToNumpyArray(this->colorShape, this->state->screenBuffer->data());
        else this->pyState->screenBuffer = pyb::none();

        if (this->state->audioBuffer != nullptr)
            this->pyState->audioBuffer = this->dataToNumpyArray(this->audioShape, this->state->audioBuffer->data());
        else this->pyState->audioBuffer = pyb::none();

        if (this->state->depthBuffer != nullptr)
            this->pyState->depthBuffer = this->dataToNumpyArray(this->grayShape, this->state->depthBuffer->data());
        else this->pyState->depthBuffer = pyb::none();

        if (this->state->labelsBuffer != nullptr) {
            this->pyState->labelsBuffer = this->dataToNumpyArray(this->grayShape, this->state->labelsBuffer->data());

            /* Update labels */
            this->pyState->labels = DoomGamePython::vectorToPyList<Label>(this->state->labels);
        }  else {
            this->pyState->labelsBuffer = pyb::none();
            this->pyState->labels = pyb::list();
        }

        if (this->state->automapBuffer != nullptr)
            this->pyState->automapBuffer = this->dataToNumpyArray(this->colorShape, this->state->automapBuffer->data());
        else this->pyState->automapBuffer = pyb::none();

        /* Updates vars */
        if (!this->state->gameVariables.empty()) {
            // Numpy array version
            this->variablesShape[0] = this->state->gameVariables.size();
            this->pyState->gameVariables = dataToNumpyArray(this->variablesShape, this->state->gameVariables.data());

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

    void DoomGamePython::newEpisode(std::string filePath) {
        ReleaseGIL gil = ReleaseGIL();  // this prevents the deadlock during the start of multiplayer game, if different Doom instances are started from different Python threads
        DoomGame::newEpisode(filePath);
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
                this->colorShape.resize(3);
                this->colorShape[0] = channels;
                this->colorShape[1] = height;
                this->colorShape[2] = width;
                break;

            case GRAY8:
            case DOOM_256_COLORS8:
                this->colorShape.resize(2);
                this->colorShape[0] = height;
                this->colorShape[1] = width;
                break;

            default:
                this->colorShape.resize(3);
                this->colorShape[0] = height;
                this->colorShape[1] = width;
                this->colorShape[2] = channels;
        }

        this->grayShape[0] = height;
        this->grayShape[1] = width;

        this->audioShape[0] = this->getAudioSamplesPerTic() * this->getAudioBufferSize();
        this->audioShape[1] = 2;
    }


    template<class T> pyb::list DoomGamePython::vectorToPyList(const std::vector<T>& vector){
        pyb::list pyList;
        for (auto& i : vector) pyList.append(i);
        return pyList;
    }

    template<class T> std::vector<T> DoomGamePython::pyListToVector(pyb::list const &pyList){
        size_t pyLen = pyb::len(pyList);
        std::vector<T> vector = std::vector<T>(pyLen);
        for (size_t i = 0; i < pyLen; ++i) vector[i] = pyb::cast<T>(pyList[i]);
        return vector;
    }

    template<class T> std::vector<T> DoomGamePython::pyArrayToVector(pyb::array_t<T> const &pyArray){
        if (pyArray.ndim() != 1)
            throw std::runtime_error("Number of dimensions larger than 1, should be 1D ndarray");

        size_t pyLen = pyArray.shape(0);
        std::vector<T> vector = std::vector<T>(pyLen);
        for (size_t i = 0; i < pyLen; ++i) vector[i] = pyArray.at(i);
        return vector;
    }

    template<typename T> std::vector<T> DoomGamePython::pyObjectToVector(pyb::object const &pyObject) {
        if(pyb::isinstance<pyb::list>(pyObject) || pyb::isinstance<pyb::tuple>(pyObject))
            return pyListToVector<T>(pyObject);
        else if(pyb::isinstance<pyb::array>(pyObject))
            return pyArrayToVector<T>(pyObject);
        else throw std::runtime_error("Unsupported type, should be list or 1D ndarray of numeric or boolean values");
    }

    template<class T> pyb::array_t<T> DoomGamePython::dataToNumpyArray(std::vector<pyb::ssize_t> dims, T *data){
        return pyb::array(dims, data);
    }
}
