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

#define LUABIND_BUILDING

#include <luabind/class.hpp>
#include <luabind/detail/garbage_collector.hpp>
#include <luabind/get_main_thread.hpp>
#include <luabind/set_package_preload.hpp>

#include <luabind/lua_include.hpp>


namespace luabind {

namespace
{

  int make_property(lua_State* L)
  {
      int args = lua_gettop(L);

      if (args == 0 || args > 2)
      {
          lua_pushliteral(L, "make_property() called with wrong number of arguments.");
          lua_error(L);
      }

      if (args == 1)
          lua_pushnil(L);

      lua_pushcclosure(L, &detail::property_tag, 2);
      return 1;
  }

  int main_thread_tag;

  int deprecated_super(lua_State* L)
  {
      lua_pushliteral(L,
          "DEPRECATION: 'super' has been deprecated in favor of "
          "directly calling the base class __init() function. "
          "This error can be disabled by calling 'luabind::disable_super_deprecation()'."
      );
      lua_error(L);

      return 0;
  }

} // namespace unnamed

    LUABIND_API lua_State* get_main_thread(lua_State* L)
    {
        lua_pushlightuserdata(L, &main_thread_tag);
        lua_rawget(L, LUA_REGISTRYINDEX);
        lua_State* result = static_cast<lua_State*>(lua_touserdata(L, -1));
        lua_pop(L, 1);

        if (!result)
            throw std::runtime_error("Unable to get main thread, luabind::open() not called?");

        return result;
    }
    namespace {
        template<typename T>
        inline void* create_gc_udata(lua_State* L, void* tag) {
            void* storage = lua_newuserdata(L, sizeof(T));

            // set gc metatable
            lua_createtable(L, 0, 1); // one (non-sequence) element.
            lua_pushcfunction(L, &detail::garbage_collector<T>);
            lua_setfield(L, -2, "__gc");
            lua_setmetatable(L, -2);

            lua_rawsetp(L, LUA_REGISTRYINDEX, tag);
            return storage;
        }

        template<typename T>
        inline void push_gc_udata(lua_State* L, void* tag) {
            void* storage = create_gc_udata<T>(L, tag);
            new (storage) T;
        }

        template<typename T, typename A1>
        inline void push_gc_udata(lua_State* L, void* tag, A1 constructorArg) {
            void* storage = create_gc_udata<T>(L, tag);
            new (storage) T(constructorArg);
        }
    }

    LUABIND_API void open(lua_State* L)
    {
        bool is_main_thread = lua_pushthread(L) == 1;
        lua_pop(L, 1);

        if (!is_main_thread)
        {
            throw std::runtime_error(
                "luabind::open() must be called with the main thread "
                "lua_State*"
            );
        }

        push_gc_udata<detail::class_registry>(L, &detail::classes_tag, L);
        push_gc_udata<detail::class_id_map>(L, &detail::classid_map_tag);
        push_gc_udata<detail::cast_graph>(L, &detail::cast_graph_tag);
        push_gc_udata<detail::class_map>(L, &detail::class_map_tag);

        // add functions (class, cast etc...)
        lua_pushcfunction(L, detail::create_class::stage1);
        lua_setglobal(L, "class");

        lua_pushcfunction(L, &make_property);
        lua_setglobal(L, "property");

        lua_pushlightuserdata(L, &main_thread_tag);
        lua_pushlightuserdata(L, L);
        lua_rawset(L, LUA_REGISTRYINDEX);

        lua_pushcfunction(L, &deprecated_super);
        lua_setglobal(L, "super");
    }

} // namespace luabind
