// Copyright Daniel Wallin 2007. Use, modification and distribution is
// subject to the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef LUABIND_ITERATOR_POLICY__071111_HPP
# define LUABIND_ITERATOR_POLICY__071111_HPP

# include <luabind/config.hpp>           // for LUABIND_ANONYMOUS_FIX
# include <luabind/detail/convert_to_lua.hpp>  // for convert_to_lua
# include <luabind/detail/policy.hpp>    // for index_map, etc

# include <new>                          // for operator new

namespace luabind {
	namespace detail {

		struct null_type;

		template <class Iterator>
		struct iterator
		{
			static int next(lua_State* L)
			{
				iterator* self = static_cast<iterator*>(
					lua_touserdata(L, lua_upvalueindex(1)));

				if(self->first != self->last)
				{
					push_to_lua(L, *self->first);
					++self->first;
				}
				else
				{
					lua_pushnil(L);
				}

				return 1;
			}

			static int destroy(lua_State* L)
			{
				iterator* self = static_cast<iterator*>(lua_touserdata(L, 1));
				self->~iterator();
				return 0;
			}

			iterator(Iterator first, Iterator last)
				: first(first)
				, last(last)
			{}

			Iterator first;
			Iterator last;
		};

		template <class Iterator>
		int make_range(lua_State* L, Iterator first, Iterator last)
		{
			void* storage = lua_newuserdata(L, sizeof(iterator<Iterator>));
			lua_newtable(L);
			lua_pushcclosure(L, iterator<Iterator>::destroy, 0);
			lua_setfield(L, -2, "__gc");
			lua_setmetatable(L, -2);
			lua_pushcclosure(L, iterator<Iterator>::next, 1);
			new (storage) iterator<Iterator>(first, last);
			return 1;
		}

		template <class Container>
		int make_range(lua_State* L, Container& container)
		{
			return make_range(L, container.begin(), container.end());
		}

		struct iterator_converter
		{
			using type = iterator_converter;

			template <class Container>
			void to_lua(lua_State* L, Container& container)
			{
				make_range(L, container);
			}

			template <class Container>
			void tu_lua(lua_State* L, Container const& container)
			{
				make_range(L, container);
			}
		};

		struct iterator_policy
		{
			template <class T, class Direction>
			struct specialize
			{
				static_assert(std::is_same<Direction, cpp_to_lua>::value, "Iterator policy can only convert from cpp to lua.");
				using type = iterator_converter;
			};
		};

	} // namespace detail
} // namespace luabind

namespace luabind {

	using return_stl_iterator = policy_list<converter_policy_injector<0, detail::iterator_policy>>;

} // namespace luabind

#endif // LUABIND_ITERATOR_POLICY__071111_HPP

