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

#ifndef LUABIND_CALL_FUNCTION_HPP_INCLUDED
#define LUABIND_CALL_FUNCTION_HPP_INCLUDED

#include <luabind/config.hpp>

#include <luabind/error.hpp>
#include <luabind/detail/convert_to_lua.hpp>
#include <luabind/detail/pcall.hpp>
#include <luabind/detail/call_shared.hpp>
#include <luabind/detail/stack_utils.hpp>

namespace luabind
{
	namespace adl {
		class object;
	}

	using adl::object;

	namespace detail {

		template< typename PolicyList, unsigned int pos >
		void push_arguments(lua_State* /*L*/) {}

		template< typename PolicyList, unsigned int Pos, typename Arg0, typename... Args >
		void push_arguments(lua_State* L, Arg0&& arg0, Args&&... args)
		{
			using converter_type = specialized_converter_policy< fetched_converter_policy<Pos, PolicyList>, Arg0, cpp_to_lua >;
			converter_type().to_lua(L, unwrapped<Arg0>::get(std::forward<Arg0>(arg0)));
			push_arguments<PolicyList, Pos + 1>(L, std::forward<Args>(args)...);
		}

#ifndef LUABIND_NO_INTERNAL_TAG_ARGUMENTS
		template<typename Ret, typename PolicyList, typename... Args, unsigned int... Indices, typename Fn>
		void call_function_impl(lua_State* L, int m_params, Fn fn, std::true_type /* void */, meta::index_list<Indices...>, Args&&... args)
		{
			int top = lua_gettop(L);

			push_arguments<PolicyList, 1>(L, std::forward<Args>(args)...);

			if(fn(L, sizeof...(Args), 0)) {
				assert(lua_gettop(L) == top - m_params + 1);
				call_error(L);
			}
			// pops the return values from the function call
			stack_pop pop(L, lua_gettop(L) - top + m_params);
		}

		template<typename Ret, typename PolicyList, typename... Args, unsigned int... Indices, typename Fn>
		Ret call_function_impl(lua_State* L, int m_params, Fn fn, std::false_type /* void */, meta::index_list<Indices...>, Args&&... args)
		{
			int top = lua_gettop(L);

			push_arguments<PolicyList, 1>(L, std::forward<Args>(args)...);

			if(fn(L, sizeof...(Args), 1)) {
				assert(lua_gettop(L) == top - m_params + 1);
				call_error(L);
			}
			// pops the return values from the function call
			stack_pop pop(L, lua_gettop(L) - top + m_params);

			specialized_converter_policy_n<0, PolicyList, Ret, lua_to_cpp> converter;
			if(converter.match(L, decorated_type<Ret>(), -1) < 0) {
				cast_error<Ret>(L);
			}

			return converter.to_cpp(L, decorated_type<Ret>(), -1);
		}
#else
		template<typename Ret, typename PolicyList, typename IndexList, unsigned int NumParams, int(*Function)(lua_State*, int, int), bool IsVoid = std::is_void<Ret>::value>
		struct call_function_struct;

		template<typename Ret, typename PolicyList, unsigned int NumParams, int(*Function)(lua_State*, int, int), unsigned int... Indices >
		struct call_function_struct< Ret, PolicyList, meta::index_list<Indices...>, NumParams, Function, true /* void */ >
		{
			template< typename... Args >
			static void call(lua_State* L, Args&&... args) {
				int top = lua_gettop(L);

				push_arguments<PolicyList, 1>(L, std::forward<Args>(args)...);

				if(Function(L, sizeof...(Args), 0)) {
					assert(lua_gettop(L) == int(top - NumParams + 1));
					call_error(L);
				}
				// pops the return values from the function call
				stack_pop pop(L, lua_gettop(L) - top + NumParams);
			}
		};

		template<typename Ret, typename PolicyList, unsigned int NumParams, int(*Function)(lua_State*, int, int), unsigned int... Indices >
		struct call_function_struct< Ret, PolicyList, meta::index_list<Indices...>, NumParams, Function, false /* void */ >
		{
			template< typename... Args >
			static Ret call(lua_State* L, Args&&... args) {
				int top = lua_gettop(L);

				push_arguments<PolicyList, 1>(L, std::forward<Args>(args)...);

				if(Function(L, sizeof...(Args), 1)) {
					assert(lua_gettop(L) == top - NumParams + 1);
					call_error(L);
				}
				// pops the return values from the function call
				stack_pop pop(L, lua_gettop(L) - top + NumParams);

				specialized_converter_policy_n<0, PolicyList, Ret, lua_to_cpp> converter;
				if(converter.match(L, decorated_type<Ret>(), -1) < 0) {
					cast_error<Ret>(L);
				}

				return converter.to_cpp(L, decorated_type<Ret>(), -1);
			}
		};
#endif
	}

	template<class R, typename PolicyList = no_policies, typename... Args>
	R call_pushed_function(lua_State* L, Args&&... args)
	{
#ifndef LUABIND_NO_INTERNAL_TAG_ARGUMENTS
		return detail::call_function_impl<R, PolicyList>(L, 1, &detail::pcall, std::is_void<R>(), meta::index_range<1, sizeof...(Args)+1>(), std::forward<Args>(args)...);
#else
		return detail::call_function_struct<R, PolicyList, meta::index_range<1, sizeof...(Args)+1>, 1, &detail::pcall >::call(L, std::forward<Args>(args)...);
#endif
	}

	template<class R, typename PolicyList = no_policies, typename... Args>
	R call_function(lua_State* L, const char* name, Args&&... args)
	{
		assert(name && "luabind::call_function() expects a function name");
		lua_getglobal(L, name);
		return call_pushed_function<R, PolicyList>(L, std::forward<Args>(args)...);
	}

	template<class R, typename PolicyList = no_policies, typename... Args>
	R resume_pushed_function(lua_State* L, Args&&... args)
	{
#ifndef LUABIND_NO_INTERNAL_TAG_ARGUMENTS
		return detail::call_function_impl<R, PolicyList>(L, 1, &detail::resume_impl, std::is_void<R>(), meta::index_range<1, sizeof...(Args)+1>(), std::forward<Args>(args)...);
#else
		return detail::call_function_struct<R, PolicyList, meta::index_range<1, sizeof...(Args)+1>, 1, &detail::resume_impl >::call(L, std::forward<Args>(args)...);
#endif
	}

	template<class R, typename PolicyList = no_policies, typename... Args>
	R resume_function(lua_State* L, const char* name, Args&&... args)
	{
		assert(name && "luabind::resume_function() expects a function name");
		lua_getglobal(L, name);
		return resume_pushed_function<R, PolicyList>(L, std::forward<Args>(args)...);
	}

	template<class R, typename PolicyList = no_policies, typename... Args>
	R resume(lua_State* L, Args&&... args)
	{
#ifndef LUABIND_NO_INTERNAL_TAG_ARGUMENTS
		return detail::call_function_impl<R, PolicyList>(L, 0, &detail::resume_impl, std::is_void<R>(), meta::index_range<1, sizeof...(Args)+1>(), std::forward<Args>(args)...);
#else
		return detail::call_function_struct<R, PolicyList, meta::index_range<1, sizeof...(Args)+1>, 0, &detail::resume_impl >::call(L, std::forward<Args>(args)...);
#endif
	}

}

#endif // LUABIND_CALL_FUNCTION_HPP_INCLUDED

