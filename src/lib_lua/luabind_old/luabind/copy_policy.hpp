// Copyright Daniel Wallin 2008. Use, modification and distribution is
// subject to the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef LUABIND_COPY_POLICY_081021_HPP
# define LUABIND_COPY_POLICY_081021_HPP

# include <luabind/detail/policy.hpp>

namespace luabind {
	namespace detail {

		struct copy_converter
		{
			template <class T>
			void to_lua(lua_State* L, T const& x)
			{
				value_converter().to_lua(L, x);
			}

			template <class T>
			void to_lua(lua_State* L, T* x)
			{
				if(!x)
					lua_pushnil(L);
				else
					to_lua(L, *x);
			}
		};

		struct copy_policy
		{
			template <class T, class Direction>
			struct specialize
			{
				static_assert(std::is_same<Direction, cpp_to_lua>::value, "Copy policy only supports cpp -> lua");
				using type = copy_converter;
			};
		};

	} // namespace detail

		// Caution: If we use the aliased type "policy_list" here, MSVC crashes.
	template< unsigned int N >
	using copy_policy = meta::type_list< converter_policy_injector< N, detail::copy_policy > >;

} // namespace luabind

#endif // LUABIND_COPY_POLICY_081021_HPP

