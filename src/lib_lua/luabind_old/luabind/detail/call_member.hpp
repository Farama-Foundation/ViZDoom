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

#ifndef LUABIND_CALL_MEMBER_HPP_INCLUDED
#define LUABIND_CALL_MEMBER_HPP_INCLUDED

#include <luabind/config.hpp>
#include <luabind/detail/convert_to_lua.hpp>
#include <luabind/detail/pcall.hpp>
#include <luabind/error.hpp>
#include <luabind/detail/stack_utils.hpp>
#include <luabind/detail/call_shared.hpp>
#include <luabind/object.hpp>

namespace luabind
{
	using adl::object;

	namespace detail {

		template<class R, typename PolicyList, unsigned int... Indices, typename... Args>
		R call_member_impl(lua_State* L, std::true_type /*void*/, meta::index_list<Indices...>, Args&&... args)
		{
			// don't count the function and self-reference
			// since those will be popped by pcall
			int top = lua_gettop(L) - 2;

			// pcall will pop the function and self reference
			// and all the parameters

			meta::init_order{ (
				specialized_converter_policy_n<Indices, PolicyList, typename unwrapped<Args>::type, cpp_to_lua>().to_lua(L, unwrapped<Args>::get(std::forward<Args>(args))), 0)...
			};

			if(pcall(L, sizeof...(Args)+1, 0))
			{
				assert(lua_gettop(L) == top + 1);
				call_error(L);
			}
			// pops the return values from the function
			stack_pop pop(L, lua_gettop(L) - top);
		}

		template<class R, typename PolicyList, unsigned int... Indices, typename... Args>
		R call_member_impl(lua_State* L, std::false_type /*void*/, meta::index_list<Indices...>, Args&&... args)
		{
			// don't count the function and self-reference
			// since those will be popped by pcall
			int top = lua_gettop(L) - 2;

			// pcall will pop the function and self reference
			// and all the parameters

			meta::init_order{ (
				specialized_converter_policy_n<Indices, PolicyList, typename unwrapped<Args>::type, cpp_to_lua>().to_lua(L, unwrapped<Args>::get(std::forward<Args>(args))), 0)...
			};

			if(pcall(L, sizeof...(Args)+1, 1))
			{
				assert(lua_gettop(L) == top + 1);
				call_error(L);
			}
			// pops the return values from the function
			stack_pop pop(L, lua_gettop(L) - top);

			specialized_converter_policy_n<0, PolicyList, R, lua_to_cpp> converter;
			if(converter.match(L, decorated_type<R>(), -1) < 0) {
				cast_error<R>(L);
			}

			return converter.to_cpp(L, decorated_type<R>(), -1);
		}


	} // detail

	template<class R, typename PolicyList = no_policies, typename... Args>
	R call_member(object const& obj, const char* name, Args&&... args)
	{
		// this will be cleaned up by the proxy object
		// once the call has been made

		// get the function
		obj.push(obj.interpreter());
		lua_pushstring(obj.interpreter(), name);
		lua_gettable(obj.interpreter(), -2);
		// duplicate the self-object
		lua_pushvalue(obj.interpreter(), -2);
		// remove the bottom self-object
		lua_remove(obj.interpreter(), -3);

		// now the function and self objects
		// are on the stack. These will both
		// be popped by pcall

		return detail::call_member_impl<R, PolicyList>(obj.interpreter(), std::is_void<R>(), meta::index_range<1, sizeof...(Args)+1>(), std::forward<Args>(args)...);
	}

	template <class R, typename... Args>
	R call_member(wrap_base const* self, char const* fn, Args&&... args)
	{
		return self->call<R>(fn, std::forward<Args>(args)...);
	}

}

#endif // LUABIND_CALL_MEMBER_HPP_INCLUDED

