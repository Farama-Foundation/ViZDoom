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

#ifndef LUABIND_RETURN_REFERENCE_TO_POLICY_HPP_INCLUDED
#define LUABIND_RETURN_REFERENCE_TO_POLICY_HPP_INCLUDED

#include <luabind/detail/policy.hpp>    // for index_map, policy_cons, etc
#include <luabind/lua_include.hpp>      // for lua_State, lua_pushnil, etc

namespace luabind {
	namespace detail {

		struct cpp_to_lua;
		struct null_type;

		template<class T>
		struct return_reference_to_converter;

		template<>
		struct return_reference_to_converter<cpp_to_lua>
		{
			template<class T>
			void to_lua(lua_State* L, const T&)
			{
				lua_pushnil(L);
			}
		};

		template< unsigned int N >
		struct return_reference_to_policy : detail::converter_policy_has_postcall_tag
		{
			template<typename StackIndexList>
			static void postcall(lua_State* L, int results, StackIndexList)
			{
				lua_pushvalue(L, meta::get<StackIndexList, N>::value);
				lua_replace(L, meta::get<StackIndexList, 0>::value + results);
			}

			template<class T, class Direction>
			struct specialize
			{
				using type = return_reference_to_converter<Direction>;
			};
		};

	}

	template<unsigned int N>
	using return_reference_to = meta::type_list<converter_policy_injector<0, detail::return_reference_to_policy<N>>>;

}

#endif // LUABIND_RETURN_REFERENCE_TO_POLICY_HPP_INCLUDED

