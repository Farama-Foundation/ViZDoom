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

#ifndef LUABIND_CONVERSION_BASE_HPP_INCLUDED
#define LUABIND_CONVERSION_BASE_HPP_INCLUDED

#include <type_traits>
#include <luabind/lua_include.hpp>
#include <luabind/detail/decorate_type.hpp>
#include <luabind/detail/make_instance.hpp>
#include <luabind/pointer_traits.hpp>
#include <luabind/from_stack.hpp>


namespace luabind {
	namespace detail {

		// Something's strange with the references here... need to know when to copy :(
		template <class T, class Clone>
		void make_pointee_instance(lua_State* L, T&& x, std::true_type, Clone)
		{
			if(get_pointer(x))
			{
				make_pointer_instance(L, std::forward<T>(x));
			}
			else
			{
				lua_pushnil(L);
			}
		}

		template <class T>
		void make_pointee_instance(lua_State* L, T&& x, std::false_type, std::true_type)
		{
			using value_type = typename std::remove_reference<T>::type;

			std::unique_ptr<value_type> ptr(new value_type(std::move(x)));
			make_pointer_instance(L, std::move(ptr));
		}

		template <class T>
		void make_pointee_instance(lua_State* L, T&& x, std::false_type, std::false_type)
		{
			// need a second make_instance that moves x into place
			make_pointer_instance(L, &x);
		}

		template <class T, class Clone>
		void make_pointee_instance(lua_State* L, T&& x, Clone)
		{
			make_pointee_instance(L, std::forward<T>(x), has_get_pointer<T>(), Clone());
		}

	}

	template <class T, class Enable>
	struct default_converter;

}

#endif

