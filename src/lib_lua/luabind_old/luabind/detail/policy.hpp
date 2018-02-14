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


#ifndef LUABIND_POLICY_HPP_INCLUDED
#define LUABIND_POLICY_HPP_INCLUDED

#include <luabind/config.hpp>
#include <luabind/detail/primitives.hpp>
#include <luabind/detail/decorate_type.hpp>

#include <typeinfo>
#include <type_traits>
#include <string>
#include <memory>

#include <luabind/detail/meta.hpp>

#if LUA_VERSION_NUM < 502
# define lua_rawlen lua_objlen
#endif

namespace luabind
{
	template< typename... T >
	using policy_list = meta::type_list< T... >;
	using no_policies = policy_list< >;

	namespace detail {

		struct converter_policy_has_postcall_tag {};

	}

	// A converter policy injector instructs the call mechanism to use a certain converter policy for
	// an element of a function call signature that is denoted by the parameter Index
	// 0 stands for the return value, while 1 might denote an implicit "this" argument or the first
	// actual argument of the function call.
	template< unsigned int Index, typename T >
	struct converter_policy_injector
	{
		enum { has_postcall = std::is_convertible<T, detail::converter_policy_has_postcall_tag >::value };
	};

	// A call policy injector instructs the call mechanism to call certain static function "postcall" on type T
	// after having executed a call.
	template< typename T >
	struct call_policy_injector
	{};


	template< typename T, typename Enable = void >
	struct default_converter;

	namespace detail
	{
		struct default_policy
		{
			template<class T, class Direction>
			struct specialize
			{
				using type = default_converter<T>;
			};
		};

		template<class T>
		struct is_primitive
			: default_converter<T>::is_native
		{};

		namespace policy_detail {
			template< unsigned int Index, typename PoliciesList >
			struct get_converter_policy;

			template< unsigned int Index, typename Injector0, typename... Injectors >
			struct get_converter_policy< Index, meta::type_list< Injector0, Injectors... > >
			{
				using type = typename get_converter_policy< Index, meta::type_list< Injectors... > >::type;
			};

			template< unsigned int Index, typename ConverterPolicy, typename... Injectors >
			struct get_converter_policy< Index, meta::type_list< converter_policy_injector< Index, ConverterPolicy >, Injectors... > >
			{
				using type = ConverterPolicy;
			};

			template< unsigned int Index >
			struct get_converter_policy< Index, meta::type_list< > >
			{
				using type = default_policy;
			};
		}

		// Fetches converter policy for Signature element [Index] from policy list [PolicyList]
		template<unsigned int Index, typename PolicyList >
		using fetched_converter_policy = typename policy_detail::get_converter_policy<Index, PolicyList>::type;

		// Specializes converter policy [ConverterPolicy] for type [Type] in direction [Direction]
		template<typename ConverterPolicy, typename Type, typename Direction>
		using specialized_converter_policy = typename ConverterPolicy::template specialize<Type, Direction>::type;

		// Fetches the converter policy for Signature element [Index] from the policy list [PolicyList] and specializes it
		// for the concrete type [T] with [Direction] being either "lua_to_cpp" or "cpp_to_lua".
		template<unsigned int Index, typename PolicyList, typename T, typename Direction>
		using specialized_converter_policy_n = typename policy_detail::get_converter_policy<Index, PolicyList>::type::template specialize<T, Direction >::type;

		/*
			call_policies
		*/

		template< typename List, class Sought >
		struct has_call_policy : public meta::contains< List, call_policy_injector< Sought > >
		{
		};

	}
} // namespace luabind::detail

namespace luabind {
	extern LUABIND_API meta::index<0> return_value;
	extern LUABIND_API meta::index<0> result;
	extern LUABIND_API meta::index<1> _1;
	extern LUABIND_API meta::index<2> _2;
	extern LUABIND_API meta::index<3> _3;
}

#endif // LUABIND_POLICY_HPP_INCLUDED

