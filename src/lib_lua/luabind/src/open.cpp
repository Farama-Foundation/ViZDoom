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

#include <luabind/lua_include.hpp>

#include <luabind/class.hpp>
#include <luabind/get_main_thread.hpp>
#include <luabind/set_package_preload.hpp>
#include <luabind/function_introspection.hpp>
#include <luabind/detail/garbage_collector.hpp>

namespace luabind {

namespace
{

  int make_property(lua_State* L)
  {
      int args = lua_gettop(L);

      if (args == 0 || args > 2)
      {
          lua_pushstring(L, "make_property() called with wrong number of arguments.");
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
      lua_pushstring(L,
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
        inline void * shared_create_userdata(lua_State* L, const char * name) {
            lua_pushstring(L, name);
            void* storage = lua_newuserdata(L, sizeof(T));

            // set gc metatable
            lua_newtable(L);
            lua_pushcclosure(L, &detail::garbage_collector<T>, 0);
            lua_setfield(L, -2, "__gc");
            lua_setmetatable(L, -2);

            lua_settable(L, LUA_REGISTRYINDEX);
            return storage;
        }

        template<typename T>
        inline void createGarbageCollectedRegistryUserdata(lua_State* L, const char * name) {
            void * storage = shared_create_userdata<T>(L, name);
            // placement "new"
            new (storage) T;
        }

        template<typename T, typename A1>
        inline void createGarbageCollectedRegistryUserdata(lua_State* L, const char * name, A1 constructorArg) {
            void * storage = shared_create_userdata<T>(L, name);

            // placement "new"
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

        createGarbageCollectedRegistryUserdata<detail::class_registry>(L, "__luabind_classes", L);
        createGarbageCollectedRegistryUserdata<detail::class_id_map>(L, "__luabind_class_id_map");
        createGarbageCollectedRegistryUserdata<detail::cast_graph>(L, "__luabind_cast_graph");
        createGarbageCollectedRegistryUserdata<detail::class_map>(L, "__luabind_class_map");

        // add functions (class, cast etc...)
        lua_pushcclosure(L, detail::create_class::stage1, 0);
        lua_setglobal(L, "class");

        lua_pushcclosure(L, &make_property, 0);
        lua_setglobal(L, "property");

        lua_pushlightuserdata(L, &main_thread_tag);
        lua_pushlightuserdata(L, L);
        lua_rawset(L, LUA_REGISTRYINDEX);

        lua_pushcclosure(L, &deprecated_super, 0);
        lua_setglobal(L, "super");

        //set_package_preload(L, "luabind.function_introspection", &bind_function_introspection);
    }

} // namespace luabind

