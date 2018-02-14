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

#ifndef LUABIND_ENUM_CONVERTER_HPP_INCLUDED
#define LUABIND_ENUM_CONVERTER_HPP_INCLUDED

#include <type_traits>
#include <luabind/detail/typetraits.hpp>
#include <luabind/detail/conversion_policies/conversion_base.hpp>

namespace luabind {
	namespace detail {

		struct enum_converter
		{
			using type      = enum_converter;
			using is_native = std::false_type;

			enum { consumed_args = 1 };

			void to_lua(lua_State* L, int val)
			{
				lua_pushnumber(L, val);
			}

			template<class T>
			T to_cpp(lua_State* L, by_value<T>, int index)
			{
				return static_cast<T>(static_cast<int>(lua_tonumber(L, index)));
			}

			template<class T>
			static int match(lua_State* L, by_value<T>, int index)
			{
				if(lua_isnumber(L, index)) {
					return 0;
				}
				else {
					return no_match;
				}
			}

			template<class T>
			T to_cpp(lua_State* L, by_const_reference<T>, int index)
			{
				return static_cast<T>(static_cast<int>(lua_tonumber(L, index)));
			}

			template<class T>
			static int match(lua_State* L, by_const_reference<T>, int index)
			{
				if(lua_isnumber(L, index)) return 0; else return no_match;
			}

			template<class T>
			void converter_postcall(lua_State*, T, int) {}
		};

	}

}

#endif

