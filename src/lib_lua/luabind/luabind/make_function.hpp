// Copyright Daniel Wallin 2008. Use, modification and distribution is
// subject to the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef LUABIND_MAKE_FUNCTION_081014_HPP
# define LUABIND_MAKE_FUNCTION_081014_HPP

# include <luabind/config.hpp>
# include <luabind/detail/call.hpp>
# include <luabind/detail/deduce_signature.hpp>
# include <luabind/detail/format_signature.hpp>
# include <luabind/exception_handler.hpp>
# include <luabind/object.hpp>

namespace luabind {

namespace detail
{
  LUABIND_API bool is_luabind_function(lua_State* L, int index);

// MSVC complains about member being sensitive to alignment (C4121)
// when F is a pointer to member of a class with virtual bases.
# ifdef BOOST_MSVC
#  pragma pack(push)
#  pragma pack(16)
# endif

  template <class F, class Signature, class Policies>
  struct function_object_impl : function_object
  {
      function_object_impl(F f_, Policies const& policies_)
        : function_object(&entry_point)
        , f(f_)
        , policies(policies_)
      {}

      int call(lua_State* L, invoke_context& ctx) const
      {
          return invoke(L, *this, ctx, f, Signature(), policies);
      }

      void format_signature(lua_State* L, char const* function) const
      {
          detail::format_signature(L, function, Signature());
      }

      static int entry_point(lua_State* L)
      {
          function_object_impl const* impl =
            *static_cast<function_object_impl const**>(
                lua_touserdata(L, lua_upvalueindex(1)));

          int results = 0;
          bool error = false;

# ifndef LUABIND_NO_EXCEPTIONS
          try
#endif
          // Scope neeeded to destroy invoke_context before calling lua_error()
          {
              invoke_context ctx;
              results = invoke(
                  L, *impl, ctx, impl->f, Signature(), impl->policies);
              if (!ctx)
              {
                  ctx.format_error(L, impl);
                  error = true;
              }
          }
#ifndef LUABIND_NO_EXCEPTIONS
          catch (...)
          {
              error = true;
              handle_exception_aux(L);
          }
#endif

          if (error) {
              assert(results >= 0);
              return lua_error(L);
          }
          if (results < 0) {
              return lua_yield(L, -results - 1);
          }
          return results;
      }

      F f;
      Policies policies;
  };

# ifdef BOOST_MSVC
#  pragma pack(pop)
# endif

  LUABIND_API object make_function_aux(
      lua_State* L, function_object* impl
  );

  LUABIND_API void add_overload(object const&, char const*, object const&);

} // namespace detail

template <class F, class Signature, class Policies>
object make_function(lua_State* L, F f, Signature, Policies)
{
    return detail::make_function_aux(
        L
      , new detail::function_object_impl<F, Signature, Policies>(
            f, Policies()
        )
    );
}

template <class F>
object make_function(lua_State* L, F f)
{
    return make_function(L, f, detail::deduce_signature(f), detail::null_type());
}

} // namespace luabind

#endif // LUABIND_MAKE_FUNCTION_081014_HPP
