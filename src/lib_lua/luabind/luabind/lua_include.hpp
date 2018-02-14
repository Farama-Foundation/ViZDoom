// Copyright (c) 2003 Daniel Wallin and Arvid Norberg

// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
// ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
// TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
// PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
// SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR
// ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
// ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
// OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef LUABIND_LUA_INCLUDE_HPP_INCLUDED
#define LUABIND_LUA_INCLUDE_HPP_INCLUDED

#include <luabind/config.hpp>

#ifndef LUABIND_CPLUSPLUS_LUA
extern "C"
{
#endif

    #include "lua.h"
    #include "lauxlib.h"

#ifndef LUABIND_CPLUSPLUS_LUA
}
#endif

#if LUA_VERSION_NUM < 502

# define lua_compare(L, index1, index2, fn) fn(L, index1, index2)
# define LUA_OPEQ lua_equal
# define LUA_OPLT lua_lessthan
# define lua_rawlen lua_objlen
# define lua_pushglobaltable(L) lua_pushvalue(L, LUA_GLOBALSINDEX)
# define lua_getuservalue lua_getfenv
# define lua_setuservalue lua_setfenv
# define LUA_OK 0

namespace luabind { namespace detail {

inline bool is_relative_index(int idx)
{
    return idx < 0 && idx > LUA_REGISTRYINDEX;
}

} } // namespace apollo::detail

LUABIND_API char const* luaL_tolstring(lua_State* L, int idx, size_t* len);

inline int lua_absindex(lua_State* L, int idx)
{
    return luabind::detail::is_relative_index(idx) ?
        lua_gettop(L) + idx + 1 : idx;
}

inline void lua_rawsetp(lua_State* L, int t, void const* k)
{
    lua_pushlightuserdata(L, const_cast<void*>(k));
    lua_insert(L, -2); // Move key beneath value.
    if (luabind::detail::is_relative_index(t))
        t -= 1; // Adjust for pushed k.
    lua_rawset(L, t);
}

inline void lua_rawgetp(lua_State* L, int t, void const* k)
{
    lua_pushlightuserdata(L, const_cast<void*>(k));
    if (luabind::detail::is_relative_index(t))
        t -= 1; // Adjust for pushed k.
    lua_rawget(L, t);
}

#endif

#endif // LUABIND_LUA_INCLUDE_HPP_INCLUDED
