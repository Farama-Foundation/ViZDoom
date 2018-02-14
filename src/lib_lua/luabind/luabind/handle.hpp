// Copyright (c) 2005 Daniel Wallin and Arvid Norberg

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

#ifndef LUABIND_HANDLE_050420_HPP
#define LUABIND_HANDLE_050420_HPP

#include <luabind/value_wrapper.hpp>

#include <luabind/lua_include.hpp>

#include <cassert>

#include <algorithm>
#include <utility>

namespace luabind {

// A reference to a Lua value. Represents an entry in the
// registry table.
class handle
{
public:
    handle();
    handle(lua_State* L, int stack_index);
    handle(lua_State* main, lua_State* L, int stack_index);
    handle(handle const& other);
    ~handle();

    handle& operator=(handle const& other);
    void swap(handle& other);

    void push(lua_State* L) const;

    lua_State* interpreter() const;
    bool is_valid() const;

    void replace(lua_State* L, int stack_index);
    void pop(lua_State* L);
    void reset();

private:
    lua_State* m_interpreter;
    int m_index;
};

inline handle::handle()
  : m_interpreter(0)
  , m_index(LUA_NOREF)
{}

inline handle::handle(handle const& other)
  : m_interpreter(other.m_interpreter)
  , m_index(LUA_NOREF)
{
    if (m_interpreter == 0) return;
    other.push(m_interpreter);
    m_index = luaL_ref(m_interpreter, LUA_REGISTRYINDEX);
}

inline handle::handle(lua_State* L, int stack_index)
  : m_interpreter(L)
  , m_index(LUA_NOREF)
{
    lua_pushvalue(L, stack_index);
    m_index = luaL_ref(L, LUA_REGISTRYINDEX);
}

inline handle::handle(lua_State* main, lua_State* L, int stack_index)
  : m_interpreter(main)
  , m_index(LUA_NOREF)
{
    lua_pushvalue(L, stack_index);
    m_index = luaL_ref(L, LUA_REGISTRYINDEX);
}

inline handle::~handle()
{
    if (is_valid())
        luaL_unref(m_interpreter, LUA_REGISTRYINDEX, m_index);
}

inline handle& handle::operator=(handle const& other)
{
    handle(other).swap(*this);
    return *this;
}

inline void handle::swap(handle& other)
{
    std::swap(m_interpreter, other.m_interpreter);
    std::swap(m_index, other.m_index);
}

inline void handle::push(lua_State* L) const
{
    lua_rawgeti(L, LUA_REGISTRYINDEX, m_index);
}

inline lua_State* handle::interpreter() const
{
    return m_interpreter;
}

inline bool handle::is_valid() const
{
    if (m_index == LUA_NOREF) {
        assert(!m_interpreter);
        return false;
    } else {
        assert(m_interpreter);
        return true;
    }
}

inline void handle::replace(lua_State* L, int stack_index)
{
    lua_pushvalue(L, stack_index);
    lua_rawseti(L, LUA_REGISTRYINDEX, m_index);
}

inline void handle::reset()
{
    if (is_valid()) {
        luaL_unref(m_interpreter, LUA_REGISTRYINDEX, m_index);
        m_index = LUA_NOREF;
    }
}

inline void handle::pop(lua_State* L)
{
    reset();
    m_interpreter = L;
    m_index = luaL_ref(L, LUA_REGISTRYINDEX);
}

template<>
struct value_wrapper_traits<handle>
{
    typedef boost::mpl::true_ is_specialized;

    static lua_State* interpreter(handle const& value)
    {
        return value.interpreter();
    }

    static void unwrap(lua_State* L, handle const& value)
    {
        value.push(L);
    }

    static bool check(...)
    {
        return true;
    }
};

} // namespace luabind

#endif // LUABIND_HANDLE_050420_HPP
