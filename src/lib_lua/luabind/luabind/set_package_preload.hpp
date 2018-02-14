/** @file
    @brief Header

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

#pragma once
#ifndef INCLUDED_set_package_preload_hpp_GUID_563c882e_86f7_4ea7_8603_4594ea41737e
#define INCLUDED_set_package_preload_hpp_GUID_563c882e_86f7_4ea7_8603_4594ea41737e

// Internal Includes
#include <luabind/config.hpp>
#include <luabind/make_function.hpp>
#include <luabind/object_fwd.hpp>

// Library/third-party includes
#include <luabind/lua_state_fwd.hpp>

// Standard includes
// - none


namespace luabind {
    LUABIND_API void set_package_preload(
        lua_State* L,
        char const* modulename,
        object const& loader);

    template <typename F>
    void set_package_preload(
        lua_State* L,
        char const* modulename,
        F loader)
    {
        object f = make_function(L, loader);
        // Call add_overload to correctly set the name of f. Inefficiency
        // should not matter here.
        detail::add_overload(luabind::newtable(L), modulename, f);
        set_package_preload(L, modulename, f);
    }
}
#endif // INCLUDED_set_package_preload_hpp_GUID_563c882e_86f7_4ea7_8603_4594ea41737e
