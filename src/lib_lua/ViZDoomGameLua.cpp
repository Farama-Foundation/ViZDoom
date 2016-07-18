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
#include <iostream>

using namespace luabind;

namespace vizdoom {

    void DoomGameLua::setAction(lb::object const& lAction) {
        DoomGame::setAction(DoomGameLua::lTableToVector<int>(lAction));
    }

    double DoomGameLua::makeAction(lb::object const& lAction) {
        return DoomGame::makeAction(DoomGameLua::lTableToVector<int>(lAction));
    }

    double DoomGameLua::makeAction(lb::object const& lAction, unsigned int tics) {
        return DoomGame::makeAction(DoomGameLua::lTableToVector<int>(lAction), tics);
    }

    GameStateLua DoomGameLua::getState() {

        GameStateLua lState;
        lState.number = this->state->number;

        return lState;

    }

    template<class T> std::vector<T> DoomGameLua::lTableToVector(lb::object const& lTable){
        std::vector<T> vector;

        for (lb::iterator i(lTable), end; i != end; ++i ) {
            T val = object_cast<T>(*i);
            vector.push_back(val);
        }

        return vector;
    }

}