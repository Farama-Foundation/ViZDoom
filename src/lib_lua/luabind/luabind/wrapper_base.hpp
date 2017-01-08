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

#ifndef LUABIND_WRAPPER_BASE_HPP_INCLUDED
#define LUABIND_WRAPPER_BASE_HPP_INCLUDED

#include <luabind/config.hpp>
#include <luabind/weak_ref.hpp>
#include <luabind/detail/ref.hpp>
#include <luabind/detail/meta.hpp>
#include <type_traits>
#include <stdexcept>

namespace luabind
{
	namespace detail
	{
		struct wrap_access;

		// implements the selection between dynamic dispatch
		// or default implementation calls from within a virtual
		// function wrapper. The input is the self reference on
		// the top of the stack. Output is the function to call
		// on the top of the stack (the input self reference will
		// be popped)
		LUABIND_API void do_call_member_selection(lua_State* L, char const* name);

		template<class R, typename PolicyList = meta::type_list<>, unsigned int... Indices, typename... Args>
		R call_member_impl(lua_State* L, std::true_type /*void*/, meta::index_list<Indices...>, Args&&... args);

		template<class R, typename PolicyList = meta::type_list<>, unsigned int... Indices, typename... Args>
		R call_member_impl(lua_State* L, std::false_type /*void*/, meta::index_list<Indices...>, Args&&... args);
	}

	struct wrapped_self_t : weak_ref
	{
		detail::lua_reference m_strong_ref;
	};

	struct wrap_base
	{
		friend struct detail::wrap_access;
		wrap_base() {}

		template<class R, typename... Args>
		R call(char const* name, Args&&... args) const
		{
			// this will be cleaned up by the proxy object
			// once the call has been made

			// TODO: what happens if this virtual function is
			// dispatched from a lua thread where the state
			// pointer is different?

			// get the function
			lua_State* L = m_self.state();
			m_self.get(L);
			assert(!lua_isnil(L, -1));
			detail::do_call_member_selection(L, name);

			if(lua_isnil(L, -1))
			{
				lua_pop(L, 1);
				throw std::runtime_error("Attempt to call nonexistent function");
			}

			// push the self reference as the first parameter
			m_self.get(L);

			// now the function and self objects
			// are on the stack. These will both
			// be popped by pcall
			return detail::call_member_impl<R>(L, std::is_void<R>(), meta::index_range<1, sizeof...(Args)+1>(), std::forward<Args>(args)...);
		}

	private:
		wrapped_self_t m_self;
	};

	namespace detail
	{
		struct wrap_access
		{
			static wrapped_self_t const& ref(wrap_base const& b)
			{
				return b.m_self;
			}

			static wrapped_self_t& ref(wrap_base& b)
			{
				return b.m_self;
			}
		};
	}
}

#endif // LUABIND_WRAPPER_BASE_HPP_INCLUDED

