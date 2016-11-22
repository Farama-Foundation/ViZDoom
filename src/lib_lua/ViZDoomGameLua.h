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

#ifndef __VIZDOOM_GAME_LUA_H__
#define __VIZDOOM_GAME_LUA_H__

#include "ViZDoomGame.h"

extern "C" {
    #include "lua.h"
    #include "lualib.h"
    #include "lauxlib.h"
}

#include "luabind/luabind.hpp"

namespace vizdoom {

    namespace lb = luabind;

    struct GameStateLua {
        unsigned int number;

        lb::object gameVariables;

        lb::object screenBuffer;
        lb::object depthBuffer;
        lb::object labelsBuffer;
        lb::object automapBuffer;

        lb::object labels;
    };

    class DoomGameLua : public DoomGame {
    public:

        void setAction(lb::object const& lAction);
        double makeAction(lb::object const& lAction, unsigned int tics = 1);

        GameStateLua getState(lua_State* luaState);
        lb::object getLastAction(lua_State* luaState);

        lb::object getAvailableButtons(lua_State* luaState);
        void setAvailableButtons(lb::object const& lButtons);

        lb::object getAvailableGameVariables(lua_State* luaState);
        void setAvailableGameVariables(lb::object const& lGameVariables);


        // Luabind doesn't support C++ 11 default arguments

        void newEpisode_() { this->newEpisode(); };
        void newEpisode_str(std::string _str) { this->newEpisode(_str); };

        double makeAction_obj(lb::object const& _obj){ return this->makeAction(_obj); }
        double makeAction_obj_int(lb::object const& _obj, unsigned int _int){ return this->makeAction(_obj, _int); }

        void advanceAction_() { this->advanceAction(); }
        void advanceAction_int(unsigned int _int) { this->advanceAction(_int); }
        void advanceAction_int_bool(unsigned int _int, bool _bool) { this->advanceAction(_int, _bool); }

        void addAvailableButton_btn(Button _btn) { this->addAvailableButton(_btn); }
        void addAvailableButton_btn_int(Button _btn, unsigned int _int) { this->addAvailableButton(_btn, _int); }

        void replayEpisode_str(std::string _str) { this->replayEpisode(_str); }
        void replayEpisode_str_int(std::string _str, unsigned int _int) { this->replayEpisode(_str, _int); }

    private:
        template<class T> static std::vector<T> lTableToVector(lb::object const& lTable);
        template<class T> static lb::object vectorToLTable(lua_State* luaState, const std::vector<T>& vector);

    };
}

#endif