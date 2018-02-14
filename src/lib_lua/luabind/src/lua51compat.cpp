// Copyright Christian Neumüller 2015. Use, modification and distribution is
// subject to the Boost Software License, Version 1.0. (See accompanying
// file LICENSE.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define LUABIND_BUILDING
#include <luabind/lua_include.hpp>

#if LUA_VERSION_NUM < 502

// Taken (with slightly adapted formatting) from Lua 5.2.3:
/*
** $Id: lauxlib.c,v 1.248.1.1 2013/04/12 18:48:47 roberto Exp $
** Auxiliary functions for building Lua libraries
* Copyright (C) 1994-2013 Lua.org, PUC-Rio.
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to deal in the Software without restriction, including
* without limitation the rights to use, copy, modify, merge, publish,
* distribute, sublicense, and/or sell copies of the Software, and to
* permit persons to whom the Software is furnished to do so, subject to
* the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
LUABIND_API char const* luaL_tolstring (lua_State* L, int idx, size_t* len)
{
  if (!luaL_callmeta(L, idx, "__tostring")) {  /* no metafield? */
    switch (lua_type(L, idx)) {
      case LUA_TNUMBER:
      case LUA_TSTRING:
        lua_pushvalue(L, idx);
        break;
      case LUA_TBOOLEAN:
        lua_pushstring(L, (lua_toboolean(L, idx) ? "true" : "false"));
        break;
      case LUA_TNIL:
        lua_pushliteral(L, "nil");
        break;
      default:
        lua_pushfstring(L, "%s: %p", luaL_typename(L, idx),
                                            lua_topointer(L, idx));
        break;
    }
  }
  return lua_tolstring(L, -1, len);
}

#endif // LUA_VERSION_NUM < 502
