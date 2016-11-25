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

#include "ViZDoomGameLua.h"
#include "ViZDoomController.h"

using namespace luabind;

namespace vizdoom {

    void DoomGameLua::setAction(lb::object const& lAction) {
        DoomGame::setAction(DoomGameLua::lTableToVector<int>(lAction));
    }

    double DoomGameLua::makeAction(lb::object const& lAction, unsigned int tics) {
        return DoomGame::makeAction(DoomGameLua::lTableToVector<int>(lAction), tics);
    }

    GameStateLua DoomGameLua::getState(lua_State* luaState) {

        GameStateLua lState;
        lState.number = this->state->number;

        if (this->state->screenBuffer != nullptr) {
            lua_pushlightuserdata(luaState, this->doomController->getScreenBuffer());
            lb::object buffer(lb::from_stack(luaState, -1));
            lua_pop(luaState, 1);
            lState.screenBuffer = buffer;
        }
        if (this->state->depthBuffer != nullptr) {
            lua_pushlightuserdata(luaState, this->doomController->getDepthBuffer());
            lb::object buffer(lb::from_stack(luaState, -1));
            lua_pop(luaState, 1);
            lState.depthBuffer = buffer;
        }
        if (this->state->labelsBuffer != nullptr) {
            lua_pushlightuserdata(luaState, this->doomController->getLabelsBuffer());
            lb::object buffer(lb::from_stack(luaState, -1));
            lua_pop(luaState, 1);
            lState.labelsBuffer = buffer;
        }
        if (this->state->automapBuffer != nullptr) {
            lua_pushlightuserdata(luaState, this->doomController->getAutomapBuffer());
            lb::object buffer(lb::from_stack(luaState, -1));
            lua_pop(luaState, 1);
            lState.automapBuffer = buffer;
        }

        if (this->state->gameVariables.size() > 0) {
            lState.gameVariables = lb::newtable(luaState);

            for(size_t i = 0; i < this->state->gameVariables.size(); ++i){
                lState.gameVariables[i + 1] = this->state->gameVariables[i];
            }
        }

        if(this->state->labels.size() > 0){
            lState.labels = lb::newtable(luaState);

            for(size_t i = 0; i < this->state->labels.size(); ++i){
                lState.labels[i + 1] = this->state->labels[i];
            }
        }

        return lState;

    }

    lb::object DoomGameLua::getLastAction(lua_State* luaState){
        return DoomGameLua::vectorToLTable(luaState, this->lastAction);
    }

    lb::object DoomGameLua::getAvailableButtons(lua_State* luaState){
        return DoomGameLua::vectorToLTable(luaState, this->availableButtons);
    }

    void DoomGameLua::setAvailableButtons(lb::object const& lButtons){
        DoomGame::setAvailableButtons(DoomGameLua::lTableToVector<Button>(lButtons));
    }

    lb::object DoomGameLua::getAvailableGameVariables(lua_State* luaState){
        return DoomGameLua::vectorToLTable(luaState, this->availableGameVariables);
    }

    void DoomGameLua::setAvailableGameVariables(lb::object const& lGameVariables){
        DoomGame::setAvailableGameVariables(DoomGameLua::lTableToVector<GameVariable>(lGameVariables));
    }


    template<class T> std::vector<T> DoomGameLua::lTableToVector(lb::object const& lTable){
        std::vector<T> vector;
        for (lb::iterator i(lTable), end; i != end; ++i ) {
            T val = object_cast<T>(*i);
            vector.push_back(val);
        }
        return vector;
    }

    template<class T> lb::object DoomGameLua::vectorToLTable(lua_State* luaState, const std::vector<T>& vector){
        lb::object lTable = lb::newtable(luaState);
        for(size_t i = 0; i < vector.size(); ++i){
            lTable[i + 1] = vector[i];
        }
        return lTable;
    }

}