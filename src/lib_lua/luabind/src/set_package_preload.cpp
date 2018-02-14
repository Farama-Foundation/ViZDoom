/**
    @file
    @brief Implementation

    @date 2012

    @author
    Ryan Pavlik
    <rpavlik@iastate.edu> and <abiryan@ryand.net>
    http://academic.cleardefinition.com/
    Iowa State University Virtual Reality Applications Center
    Human-Computer Interaction Graduate Program
*/

//          Copyright Iowa State University 2012.
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

#define LUABIND_BUILDING

// Internal Includes
#include <luabind/config.hpp>           // for LUABIND_API
#include <luabind/object.hpp>
#include <luabind/set_package_preload.hpp>

// Library/third-party includes
#include <luabind/lua_include.hpp>      // for lua_pushstring, lua_rawset, etc

// Standard includes
// - none

static int do_set_package_preload(lua_State* L)
{
    // Note: Use ordinary set/get instead of the raw variants, because this
    // function should not be performance sensitive anyway.
    lua_pushglobaltable(L);
    lua_getfield(L, -1, "package");
    lua_remove(L, -2); // Remove global table.
    lua_getfield(L, -1, "preload");
    lua_remove(L, -2); // Remove package table.
    lua_insert(L, -3); // Move package.preload beneath key and value.
    lua_settable(L, -3); // package.preload[modulename] = loader
    return 0;
}

static int proxy_loader(lua_State* L)
{
    luaL_checkstring(L, 1); // First argument should be the module name.
    lua_settop(L, 1); // Ignore any other arguments.
    lua_pushvalue(L, lua_upvalueindex(1)); // Push the real loader.
    lua_insert(L, 1); // Move it beneath the argument.
    lua_call(L, 1, LUA_MULTRET); // Pops everyhing.
    return lua_gettop(L);
}


namespace luabind {
    LUABIND_API void set_package_preload(
        lua_State * L, char const* modulename, object const& loader)
    {
        loader.push(L);
        lua_pushcclosure(L, &proxy_loader, 1);
        object const proxy_ldr(from_stack(L, -1));
        lua_pop(L, 1); // pop do_load.
        lua_pushcfunction(L, &do_set_package_preload);
        // Must use object for correct popping in case of errors:
        object do_set(from_stack(L, -1));
        lua_pop(L, 1);
        do_set(modulename, proxy_ldr);
    }

} // namespace luabind
