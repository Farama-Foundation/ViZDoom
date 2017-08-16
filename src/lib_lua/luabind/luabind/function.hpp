// Copyright Daniel Wallin 2008. Use, modification and distribution is
// subject to the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef LUABIND_FUNCTION2_081014_HPP
# define LUABIND_FUNCTION2_081014_HPP

# include <luabind/make_function.hpp>
# include <luabind/scope.hpp>
# include <luabind/detail/call_function.hpp>

namespace luabind {

	namespace detail
	{

		template <class F, class PolicyInjectors>
		struct function_registration : registration
		{
			function_registration(char const* name, F f)
				: name(name)
				, f(f)
			{}

			void register_(lua_State* L) const
			{
				object fn = make_function(L, f, PolicyInjectors());
				add_overload(object(from_stack(L, -1)), name, fn);
			}

			char const* name;
			F f;
		};

		LUABIND_API bool is_luabind_function(lua_State* L, int index);

	} // namespace detail

	template <class F, typename... PolicyInjectors>
	scope def(char const* name, F f, policy_list<PolicyInjectors...> const&)
	{
		return scope(std::unique_ptr<detail::registration>(
			new detail::function_registration<F, policy_list<PolicyInjectors...>>(name, f)));
	}

	template <class F>
	scope def(char const* name, F f)
	{
		return def(name, f, no_policies());
	}

} // namespace luabind

#endif // LUABIND_FUNCTION2_081014_HPP

