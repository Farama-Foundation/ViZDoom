// Copyright Daniel Wallin 2008. Use, modification and distribution is
// subject to the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define LUABIND_BUILDING

#include <luabind/make_function.hpp>

namespace luabind { namespace detail {

namespace
{

  // A pointer to this is used as a tag value to identify functions exported
  // by luabind.
  char function_tag = 0;

  int function_destroy(lua_State* L)
  {
      function_object* fn = *static_cast<function_object**>(
          lua_touserdata(L, 1));
      delete fn;
      return 0;
  }

  void push_function_metatable(lua_State* L)
  {
      lua_rawgetp(L, LUA_REGISTRYINDEX, &function_tag);

      if (lua_istable(L, -1))
          return;

      lua_pop(L, 1);

      lua_createtable(L, 0, 1); // One non-sequence entry for __gc.

      lua_pushliteral(L, "__gc");
      lua_pushcfunction(L, &function_destroy);
      lua_rawset(L, -3);

      lua_pushvalue(L, -1);
      lua_rawsetp(L, LUA_REGISTRYINDEX, &function_tag);
  }

} // namespace unnamed

LUABIND_API bool is_luabind_function(lua_State* L, int index)
{
    if (!lua_getupvalue(L, index, 2))
        return false;
    bool result = lua_touserdata(L, -1) == &function_tag;
    lua_pop(L, 1);
    return result;
}

namespace
{

  inline bool is_luabind_function(object const& obj)
  {
      obj.push(obj.interpreter());
      bool result = detail::is_luabind_function(obj.interpreter(), -1);
      lua_pop(obj.interpreter(), 1);
      return result;
  }

} // namespace unnamed

LUABIND_API void add_overload(
    object const& context, char const* name, object const& fn)
{
    function_object* f = *touserdata<function_object*>(getupvalue(fn, 1));
    f->name = name;

    if (object overloads = context[name])
    {
        if (is_luabind_function(overloads) && is_luabind_function(fn))
        {
            f->next = *touserdata<function_object*>(getupvalue(overloads, 1));
            f->keepalive = overloads;
        }
    }

    context[name] = fn;
}

LUABIND_API object make_function_aux(lua_State* L, function_object* impl)
{
    void* storage = lua_newuserdata(L, sizeof(function_object*));
    push_function_metatable(L);
    *static_cast<function_object**>(storage) = impl;
    lua_setmetatable(L, -2);

    lua_pushlightuserdata(L, &function_tag);
    lua_pushcclosure(L, impl->entry, 2);
    stack_pop pop(L, 1);

    return object(from_stack(L, -1));
}

void invoke_context::format_error(
    lua_State* L, function_object const* overloads) const
{
    char const* function_name =
        overloads->name.empty() ? "<unknown>" : overloads->name.c_str();

    if (candidate_index == 0)
    {
        lua_pushliteral(L, "No matching overload found, candidates:");
        for (function_object const* f = overloads; f != 0; f = f->next)
        {
            lua_pushliteral(L, "\n");
            f->format_signature(L, function_name);
            lua_concat(L, 3); // Inefficient, but does not use up the stack.
        }
    }
    else
    {
        // Ambiguous
        lua_pushliteral(L, "Ambiguous, candidates:");
        for (int i = 0; i < candidate_index; ++i)
        {
            lua_pushliteral(L, "\n");
            candidates[i]->format_signature(L, function_name);
            lua_concat(L, 3); // Inefficient, but does not use up the stack.
        }
        if (additional_candidates)
        {
            BOOST_ASSERT(candidate_index == max_candidates);
            lua_pushfstring(L, "\nand %d additional overload(s) not shown",
                additional_candidates);
            lua_concat(L, 2);
        }
    }
}

}} // namespace luabind::detail
