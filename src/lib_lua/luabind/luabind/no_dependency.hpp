// Copyright Daniel Wallin 2010. Use, modification and distribution is
// subject to the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef LUABIND_NO_DEPENDENCY_100324_HPP
# define LUABIND_NO_DEPENDENCY_100324_HPP

# include <luabind/detail/policy.hpp>

namespace luabind {

namespace detail
{

  struct no_dependency_policy
  {
      static void precall(lua_State*, index_map const&)
      {}

      static void postcall(lua_State*, index_map const&)
      {}
  };

  typedef policy_cons<no_dependency_policy, null_type>
      no_dependency_node;

} // namespace detail

detail::no_dependency_node const no_dependency = {};

namespace detail
{

  inline void ignore_unused_no_dependency()
  {
      (void)no_dependency;
  }

} // namespace detail

} // namespace luabind

#endif // LUABIND_NO_DEPENDENCY_100324_HPP
