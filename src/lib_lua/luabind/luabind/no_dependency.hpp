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
			static void postcall(lua_State*, int /*results*/, meta::index_list_tag)
			{}
		};

	} // namespace detail

	using no_dependency = policy_list<call_policy_injector<detail::no_dependency_policy>>;

} // namespace luabind

#endif // LUABIND_NO_DEPENDENCY_100324_HPP

