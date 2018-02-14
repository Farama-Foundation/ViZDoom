// Copyright Daniel Wallin 2008. Use, modification and distribution is
// subject to the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef LUABIND_FUNCTION2_081014_HPP
# define LUABIND_FUNCTION2_081014_HPP

# include <luabind/detail/call_function.hpp>
# include <luabind/make_function.hpp>
# include <luabind/scope.hpp>

namespace luabind {

namespace detail
{

  template <class F, class Policies>
  struct function_registration : registration
  {
      function_registration(char const* name_, F f_, Policies const& policies_)
        : name(name_)
        , f(f_)
        , policies(policies_)
      {}

      void register_(lua_State* L) const
      {
          object fn = make_function(L, f, deduce_signature(f), policies);

          add_overload(
              object(from_stack(L, -1))
            , name
            , fn
          );
      }

      char const* name;
      F f;
      Policies policies;
  };

} // namespace detail

template <class F, class Policies>
scope def(char const* name, F f, Policies const& policies)
{
    return scope(std::auto_ptr<detail::registration>(
        new detail::function_registration<F, Policies>(name, f, policies)));
}

template <class F>
scope def(char const* name, F f)
{
    return def(name, f, detail::null_type());
}

} // namespace luabind

#endif // LUABIND_FUNCTION2_081014_HPP
