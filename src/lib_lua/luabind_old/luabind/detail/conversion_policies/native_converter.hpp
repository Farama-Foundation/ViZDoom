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

#ifndef LUABIND_NATIVE_CONVERTER_HPP_INCLUDED
#define LUABIND_NATIVE_CONVERTER_HPP_INCLUDED

#include <type_traits>
#include <string>
#include <luabind/detail/conversion_policies/conversion_base.hpp>
#include <luabind/detail/call_traits.hpp>
#include <luabind/lua_include.hpp>

#if LUA_VERSION_NUM < 502
# define lua_rawlen lua_objlen
#endif

namespace luabind {

	template <class T, class Derived = default_converter<T> >
	struct native_converter_base
	{
		using is_native  = std::true_type;
		using value_type = typename detail::call_traits<T>::value_type;
		using param_type = typename detail::call_traits<T>::param_type;

		enum { consumed_args = 1 };

		template <class U>
		void converter_postcall(lua_State*, U const&, int)
		{}

		int match(lua_State* L, detail::by_value<T>, int index)
		{
			return Derived::compute_score(L, index);
		}

		int match(lua_State* L, detail::by_value<T const>, int index)
		{
			return Derived::compute_score(L, index);
		}


		int match(lua_State* L, detail::by_const_reference<T>, int index)
		{
			return Derived::compute_score(L, index);
		}

		value_type to_cpp(lua_State* L, detail::by_value<T>, int index)
		{
			return derived().to_cpp_deferred(L, index);
		}

		value_type to_cpp(lua_State* L, detail::by_const_reference<T>, int index)
		{
			return derived().to_cpp_deferred(L, index);
		}

		void to_lua(lua_State* L, param_type value)
		{
			derived().to_lua_deferred(L, value);
		}

		Derived& derived()
		{
			return static_cast<Derived&>(*this);
		}
	};

	template <typename QualifiedT>
	struct integer_converter
		: native_converter_base<typename std::remove_const<typename std::remove_reference<QualifiedT>::type>::type>
	{
		using T          = typename std::remove_const<typename std::remove_reference<QualifiedT>::type>::type;
		using value_type = typename native_converter_base<T>::value_type;
		using param_type = typename native_converter_base<T>::param_type;

		static int compute_score(lua_State* L, int index)
		{
			return lua_type(L, index) == LUA_TNUMBER ? 0 : no_match;
		}

		static value_type to_cpp_deferred(lua_State* L, int index)
		{
			if((std::is_unsigned<value_type>::value && sizeof(value_type) >= sizeof(lua_Integer)) || (sizeof(value_type) > sizeof(lua_Integer))) {
				return static_cast<T>(lua_tonumber(L, index));
			}
			else {
				return static_cast<T>(lua_tointeger(L, index));
			}
		}

		void to_lua_deferred(lua_State* L, param_type value)
		{
			if((std::is_unsigned<value_type>::value && sizeof(value_type) >= sizeof(lua_Integer)) || (sizeof(value_type) > sizeof(lua_Integer)))
			{
				lua_pushnumber(L, value);
			}
			else {
				lua_pushinteger(L, static_cast<lua_Integer>(value));
			}
		}
	};

	template <typename QualifiedT>
	struct number_converter
		: native_converter_base<typename std::remove_const<typename std::remove_reference<QualifiedT>::type>::type>
	{
		using T          = typename std::remove_const<typename std::remove_reference<QualifiedT>::type>::type;
		using value_type = typename native_converter_base<T>::value_type;
		using param_type = typename native_converter_base<T>::param_type;

		static int compute_score(lua_State* L, int index)
		{
			return lua_type(L, index) == LUA_TNUMBER ? 0 : no_match;
		}

		static value_type to_cpp_deferred(lua_State* L, int index)
		{
			return static_cast<T>(lua_tonumber(L, index));
		}

		static void to_lua_deferred(lua_State* L, param_type value)
		{
			lua_pushnumber(L, static_cast<lua_Number>(value));
		}
	};

	template <>
	struct default_converter<bool>
		: native_converter_base<bool>
	{
		static int compute_score(lua_State* L, int index)
		{
			return lua_type(L, index) == LUA_TBOOLEAN ? 0 : no_match;
		}

		static bool to_cpp_deferred(lua_State* L, int index)
		{
			return lua_toboolean(L, index) == 1;
		}

		static void to_lua_deferred(lua_State* L, bool value)
		{
			lua_pushboolean(L, value);
		}
	};


	template <>
	struct default_converter<bool const>
		: default_converter<bool>
	{};

	template <>
	struct default_converter<bool const&>
		: default_converter<bool>
	{};

	template <>
	struct default_converter<std::string>
		: native_converter_base<std::string>
	{
		static int compute_score(lua_State* L, int index)
		{
			return lua_type(L, index) == LUA_TSTRING ? 0 : no_match;
		}

		static std::string to_cpp_deferred(lua_State* L, int index)
		{
			return std::string(lua_tostring(L, index), lua_rawlen(L, index));
		}

		static void to_lua_deferred(lua_State* L, std::string const& value)
		{
			lua_pushlstring(L, value.data(), value.size());
		}
	};

	template <>
	struct default_converter<std::string&>
		: default_converter<std::string>
	{};

	template <>
	struct default_converter<std::string const>
		: default_converter<std::string>
	{};

	template <>
	struct default_converter<std::string const&>
		: default_converter<std::string>
	{};

	template <>
	struct default_converter<char const*>
	{
		using is_native = std::true_type;

		enum { consumed_args = 1 };

		template <class U>
		static int match(lua_State* L, U, int index)
		{
			int type = lua_type(L, index);
			return (type == LUA_TSTRING || type == LUA_TNIL) ? 0 : no_match;
		}

		template <class U>
		static char const* to_cpp(lua_State* L, U, int index)
		{
			return lua_tostring(L, index);
		}

		static void to_lua(lua_State* L, char const* str)
		{
			lua_pushstring(L, str);
		}

		template <class U>
		void converter_postcall(lua_State*, U, int)
		{}
	};

	template <>
	struct default_converter<const char* const>
		: default_converter<char const*>
	{};

	template <>
	struct default_converter<const char* const&>
		: default_converter<char const*>
	{};

	template <>
	struct default_converter<const char*&>
		: default_converter<char const*>
	{};

	template <>
	struct default_converter<char*>
		: default_converter<char const*>
	{};

	template <std::size_t N>
	struct default_converter<char const[N]>
		: default_converter<char const*>
	{};

	template <std::size_t N>
	struct default_converter<char[N]>
		: default_converter<char const*>
	{};

	template <std::size_t N>
	struct default_converter <char(&)[N]>
		: default_converter<char const*>
	{};

	template <std::size_t N>
	struct default_converter <const char(&)[N]>
		: default_converter<char const*>
	{};

	template <typename T>
	struct default_converter < T, typename std::enable_if< std::is_integral<typename std::remove_reference<T>::type>::value >::type >
		: integer_converter<T>
	{
	};

	template <typename T>
	struct default_converter < T, typename std::enable_if< std::is_floating_point<typename std::remove_reference<T>::type>::value >::type >
		: number_converter<T>
	{
	};

}

#if LUA_VERSION_NUM < 502
# undef lua_rawlen
#endif

#endif

