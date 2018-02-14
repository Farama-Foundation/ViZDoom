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


#ifndef LUABIND_DEPENDENCY_POLICY_HPP_INCLUDED
#define LUABIND_DEPENDENCY_POLICY_HPP_INCLUDED

#include <luabind/config.hpp>
#include <luabind/detail/policy.hpp>    // for policy_cons, etc
#include <luabind/detail/object_rep.hpp>  // for object_rep
#include <luabind/detail/primitives.hpp>  // for null_type

namespace luabind {
	namespace detail {

		// makes A dependent on B, meaning B will outlive A.
		// internally A stores a reference to B
		template<int A, int B>
		struct dependency_policy
		{
			template< unsigned int... StackIndices >
			static void postcall(lua_State* L, int results, meta::index_list<StackIndices...>)
			{
				object_rep* nurse = static_cast<object_rep*>(lua_touserdata(L, meta::get<meta::index_list<StackIndices...>, A>::value));

				// If the nurse isn't an object_rep, just make this a nop.
				if(nurse == 0)
					return;

				nurse->add_dependency(L, meta::get<meta::index_list<StackIndices...>, B>::value);
			}
		};

		template<int B>
		struct dependency_policy<0, B>
		{
			template< unsigned int... StackIndices >
			static void postcall(lua_State* L, int results, meta::index_list<StackIndices...>)
			{
				object_rep* nurse = static_cast<object_rep*>(lua_touserdata(L, meta::get<meta::index_list<StackIndices...>, 0>::value + results));

				// If the nurse isn't an object_rep, just make this a nop.
				if(nurse == 0)
					return;

				nurse->add_dependency(L, meta::get<meta::index_list<StackIndices...>, B>::value);
			}
		};

		template<int A>
		struct dependency_policy<A, 0>
		{
			template< unsigned int... StackIndices >
			static void postcall(lua_State* L, int results, meta::index_list<StackIndices...>)
			{
				object_rep* nurse = static_cast<object_rep*>(lua_touserdata(L, meta::get<meta::index_list<StackIndices...>, A>::value));

				// If the nurse isn't an object_rep, just make this a nop.
				if(nurse == 0)
					return;

				nurse->add_dependency(L, meta::get<meta::index_list<StackIndices...>, 0>::value + results);
			}
		};

	}
}

namespace luabind
{
	// Caution: If we use the aliased type "policy_list" here, MSVC crashes.
	template<unsigned int A, unsigned int B>
	using dependency_policy = meta::type_list<call_policy_injector<detail::dependency_policy<A, B>>>;

	template<unsigned int A>
	using return_internal_reference = meta::type_list<call_policy_injector<detail::dependency_policy<0, A>>>;
}

#endif // LUABIND_DEPENDENCY_POLICY_HPP_INCLUDED

