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

#ifndef LUABIND_CONVERSION_POLICIES_HPP_INCLUDED
#define LUABIND_CONVERSION_POLICIES_HPP_INCLUDED

#include <luabind/detail/typetraits.hpp>
#include <luabind/detail/meta.hpp>
#include <luabind/detail/policy.hpp>
#include <luabind/detail/conversion_policies/conversion_base.hpp>
#include <luabind/detail/conversion_policies/enum_converter.hpp>
#include <luabind/detail/conversion_policies/pointer_converter.hpp>
#include <luabind/detail/conversion_policies/reference_converter.hpp>
#include <luabind/detail/conversion_policies/value_converter.hpp>
#include <luabind/detail/conversion_policies/lua_proxy_converter.hpp>
#include <luabind/detail/conversion_policies/native_converter.hpp>
#include <luabind/detail/conversion_policies/function_converter.hpp>
#include <luabind/shared_ptr_converter.hpp>

namespace luabind {

	template <>
	struct default_converter<lua_State*>
	{
		enum { consumed_args = 0 };

		template <class U>
		lua_State* to_cpp(lua_State* L, U, int /*index*/)
		{
			return L;
		}

		template <class U>
		static int match(lua_State*, U, int /*index*/)
		{
			return 0;
		}

		template <class U>
		void converter_postcall(lua_State*, U, int) {}
	};

	namespace detail {

		// This is the one that gets hit, if default_policy doesn't hit one of the specializations defined all over the place
		template< class T >
		struct default_converter_generator
			: public meta::select_ <
			meta::case_< is_lua_proxy_arg<T>, lua_proxy_converter<T> >,
			meta::case_< std::is_enum<typename std::remove_reference<T>::type>, enum_converter >,
			meta::case_< is_nonconst_pointer<T>, pointer_converter >,
			meta::case_< is_const_pointer<T>, const_pointer_converter >,
			meta::case_< is_nonconst_reference<T>, ref_converter >,
			meta::case_< is_const_reference<T>, const_ref_converter >,
			meta::default_< value_converter >
			> ::type
		{
		};

	}

	template <class T, class Enable>
	struct default_converter
		: detail::default_converter_generator<T>::type
	{};

}

#endif

