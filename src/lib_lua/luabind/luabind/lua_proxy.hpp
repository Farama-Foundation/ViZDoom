// Copyright (c) 2005 Daniel Wallin and Arvid Norberg

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

#ifndef LUABIND_VALUE_WRAPPER_050419_HPP
#define LUABIND_VALUE_WRAPPER_050419_HPP

#include <type_traits>

namespace luabind {

	//
	// Concept "lua_proxy"
	//

	template<class T>
	struct lua_proxy_traits
	{
		using is_specialized = std::false_type;
	};

	template<class T>
	struct is_lua_proxy_type
		: lua_proxy_traits<T>::is_specialized
	{};

	template< class T >
	struct is_lua_proxy_arg
		: std::conditional<is_lua_proxy_type<typename std::remove_const<typename std::remove_reference<T>::type>::type>::value, std::true_type, std::false_type >::type
	{};

} // namespace luabind

#endif // LUABIND_VALUE_WRAPPER_050419_HPP

