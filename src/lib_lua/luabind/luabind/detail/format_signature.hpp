// Copyright Daniel Wallin 2008. Use, modification and distribution is
// subject to the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef LUABIND_FORMAT_SIGNATURE_081014_HPP
# define LUABIND_FORMAT_SIGNATURE_081014_HPP

#include <luabind/config.hpp>
#include <luabind/lua_include.hpp>
#include <luabind/typeid.hpp>
#include <luabind/detail/meta.hpp>

namespace luabind {
	namespace adl {

		class object;
		class argument;
		template <class Base>
		struct table;

	} // namespace adl

	using adl::object;
	using adl::argument;
	using adl::table;

	namespace detail {

		LUABIND_API std::string get_class_name(lua_State* L, type_id const& i);

		template <class T, class Enable = void>
		struct type_to_string
		{
			static void get(lua_State* L)
			{
				lua_pushstring(L, get_class_name(L, typeid(T)).c_str());
			}
		};

		template <class T>
		struct type_to_string<T*>
		{
			static void get(lua_State* L)
			{
				type_to_string<T>::get(L);
				lua_pushstring(L, "*");
				lua_concat(L, 2);
			}
		};

		template <class T>
		struct type_to_string<T&>
		{
			static void get(lua_State* L)
			{
				type_to_string<T>::get(L);
				lua_pushstring(L, "&");
				lua_concat(L, 2);
			}
		};

		template <class T>
		struct type_to_string<T const>
		{
			static void get(lua_State* L)
			{
				type_to_string<T>::get(L);
				lua_pushstring(L, " const");
				lua_concat(L, 2);
			}
		};

# define LUABIND_TYPE_TO_STRING(x) \
    template <> \
    struct type_to_string<x> \
    { \
        static void get(lua_State* L) \
        { \
            lua_pushstring(L, #x); \
        } \
    };

# define LUABIND_INTEGRAL_TYPE_TO_STRING(x) \
    LUABIND_TYPE_TO_STRING(x) \
    LUABIND_TYPE_TO_STRING(unsigned x)

		LUABIND_INTEGRAL_TYPE_TO_STRING(char)
			LUABIND_INTEGRAL_TYPE_TO_STRING(short)
			LUABIND_INTEGRAL_TYPE_TO_STRING(int)
			LUABIND_INTEGRAL_TYPE_TO_STRING(long)

			LUABIND_TYPE_TO_STRING(void)
			LUABIND_TYPE_TO_STRING(bool)
			LUABIND_TYPE_TO_STRING(std::string)
			LUABIND_TYPE_TO_STRING(lua_State)

			LUABIND_TYPE_TO_STRING(luabind::object)
			LUABIND_TYPE_TO_STRING(luabind::argument)

# undef LUABIND_INTEGRAL_TYPE_TO_STRING
# undef LUABIND_TYPE_TO_STRING

			template <class Base>
		struct type_to_string<table<Base> >
		{
			static void get(lua_State* L)
			{
				lua_pushstring(L, "table");
			}
		};

		inline void format_signature_aux(lua_State*, bool, meta::type_list< >)
		{}

		template <class Signature>
		void format_signature_aux(lua_State* L, bool first, Signature)
		{
			if(!first)
				lua_pushstring(L, ",");
			type_to_string<typename meta::front<Signature>::type>::get(L);
			format_signature_aux(L, false, typename meta::pop_front<Signature>::type());
		}

		template <class Signature>
		void format_signature(lua_State* L, char const* function, Signature)
		{
			using first = typename meta::front<Signature>::type;

			type_to_string<first>::get(L);

			lua_pushstring(L, " ");
			lua_pushstring(L, function);

			lua_pushstring(L, "(");
			format_signature_aux(
				L
				, true
				, typename meta::pop_front<Signature>::type()
			);
			lua_pushstring(L, ")");
			size_t sz = meta::size<Signature>::value;
			size_t ncat = sz * 2 + 2;
			if(sz == 1)
				++ncat;
			lua_concat(L, ncat);
		}

	} // namespace detail

} // namespace luabind

#endif // LUABIND_FORMAT_SIGNATURE_081014_HPP

