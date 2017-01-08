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

#ifndef LUABIND_OBJECT_050419_HPP
#define LUABIND_OBJECT_050419_HPP

#include <tuple>

#include <luabind/nil.hpp>
#include <luabind/handle.hpp>
#include <luabind/from_stack.hpp>
#include <luabind/detail/stack_utils.hpp>
#include <luabind/detail/convert_to_lua.hpp> // REFACTOR
#include <luabind/typeid.hpp>
#include <luabind/detail/crtp_iterator.hpp>
#include <luabind/lua_proxy_interface.hpp>
#include <luabind/lua_index_proxy.hpp>
#include <luabind/lua_iterator_proxy.hpp>
#include <luabind/detail/class_rep.hpp>

#if LUA_VERSION_NUM < 502
# define lua_pushglobaltable(L) lua_pushvalue(L, LUA_GLOBALSINDEX)
#endif

namespace luabind {
	namespace adl {

		// An object holds a reference to a Lua value residing
		// in the registry.
		class object :
			public lua_proxy_interface<object>
		{
		public:
			object()
			{}

			explicit object(handle const& other)
				: m_handle(other)
			{}

			explicit object(from_stack const& stack_reference)
				: m_handle(stack_reference.interpreter, stack_reference.index)
			{
			}

			template<class T>
			object(lua_State* interpreter, T const& value)
			{
				detail::push(interpreter, value);
				detail::stack_pop pop(interpreter, 1);
				handle(interpreter, -1).swap(m_handle);
			}

			template<class T, class Policies>
			object(lua_State* interpreter, T const& value, Policies const&)
			{
				detail::push(interpreter, value, Policies());
				detail::stack_pop pop(interpreter, 1);
				handle(interpreter, -1).swap(m_handle);
			}

			void push(lua_State* interpreter) const;
			lua_State* interpreter() const;
			bool is_valid() const;

			template<class T>
			index_proxy<object> operator[](T const& key) const
			{
				return index_proxy<object>(
					*this, m_handle.interpreter(), key
					);
			}

			void swap(object& other)
			{
				m_handle.swap(other.m_handle);
			}

		private:
			handle m_handle;
		};

		inline void object::push(lua_State* interpreter) const
		{
			m_handle.push(interpreter);
		}

		inline lua_State* object::interpreter() const
		{
			return m_handle.interpreter();
		}

		inline bool object::is_valid() const
		{
			return m_handle.interpreter() != 0;
		}

	} // namespace adl

	using adl::object;

	template<>
	struct lua_proxy_traits<object>
	{
		using is_specialized = std::true_type;

		static lua_State* interpreter(object const& value)
		{
			return value.interpreter();
		}

		static void unwrap(lua_State* interpreter, object const& value)
		{
			value.push(interpreter);
		}

		static bool check(...)
		{
			return true;
		}
	};

	template<class R, typename PolicyList = no_policies, typename... Args>
	R call_function(luabind::object const& obj, Args&&... args)
	{
		obj.push(obj.interpreter());
		return call_pushed_function<R, PolicyList>(obj.interpreter(), std::forward<Args>(args)...);
	}

	template<class R, typename PolicyList = no_policies, typename... Args>
	R resume_function(luabind::object const& obj, Args&&... args)
	{
		obj.push(obj.interpreter());
		return resume_pushed_function<R, PolicyList>(obj.interpreter(), std::forward<Args>(args)...);
	}

	// declared in luabind/lua_index_proxy.hpp
	template<typename Next>
	adl::index_proxy<Next>::operator object()
	{
		detail::stack_pop pop(m_interpreter, 1);
		push(m_interpreter);
		return object(from_stack(m_interpreter, -1));
	}

	// declared in luabind/lua_proxy_interface.hpp
	template<typename ProxyType>
	template<typename PolicyList, typename... Args>
	object adl::lua_proxy_interface<ProxyType>::call(Args&&... args)
	{
		return call_function<object, PolicyList>(derived(), std::forward<Args>(args)...);
	}

	// declared in luabind/lua_proxy_interface.hpp
	template<typename ProxyType>
	template<typename... Args>
	object adl::lua_proxy_interface<ProxyType>::operator()(Args&&... args)
	{
		return call<no_policies>(std::forward<Args>(args)...);
	}

	// declared in luabind/lua_iterator_proxy.hpp
	template<class AccessPolicy>
	adl::iterator_proxy<AccessPolicy>::operator object()
	{
		lua_pushvalue(m_interpreter, m_key_index);
		AccessPolicy::get(m_interpreter, m_table_index);
		detail::stack_pop pop(m_interpreter, 1);
		return object(from_stack(m_interpreter, -1));
	}

	// declared in luabind/lua_iterator_proxy.hpp
	template<class AccessPolicy>
	object detail::basic_iterator<AccessPolicy>::key() const
	{
		return object(m_key);
	}

	namespace adl {
		// Simple value_wrapper adaptor with the sole purpose of helping with
		// overload resolution. Use this as a function parameter type instead
		// of "object" or "argument" to restrict the parameter to Lua tables.
		template <class Base = object>
		struct table : Base
		{
			table(from_stack const& stack_reference)
				: Base(stack_reference)
			{}
		};

	} // namespace adl

	using adl::table;

	template <class Base>
	struct lua_proxy_traits<adl::table<Base> >
		: lua_proxy_traits<Base>
	{
		static bool check(lua_State* L, int idx)
		{
			return lua_proxy_traits<Base>::check(L, idx) &&
				lua_istable(L, idx);
		}
	};

	inline object newtable(lua_State* interpreter)
	{
		lua_newtable(interpreter);
		detail::stack_pop pop(interpreter, 1);
		return object(from_stack(interpreter, -1));
	}

	// this could be optimized by returning a proxy
	inline object globals(lua_State* interpreter)
	{
		lua_pushglobaltable(interpreter);
		detail::stack_pop pop(interpreter, 1);
		return object(from_stack(interpreter, -1));
	}

	// this could be optimized by returning a proxy
	inline object registry(lua_State* interpreter)
	{
		lua_pushvalue(interpreter, LUA_REGISTRYINDEX);
		detail::stack_pop pop(interpreter, 1);
		return object(from_stack(interpreter, -1));
	}

	template<class ValueWrapper, class K>
	inline object gettable(ValueWrapper const& table, K const& key)
	{
		lua_State* interpreter = lua_proxy_traits<ValueWrapper>::interpreter(table);

		lua_proxy_traits<ValueWrapper>::unwrap(interpreter, table);
		detail::stack_pop pop(interpreter, 2);
		detail::push(interpreter, key);
		lua_gettable(interpreter, -2);
		return object(from_stack(interpreter, -1));
	}

	template<class ValueWrapper, class K, class T>
	inline void settable(ValueWrapper const& table, K const& key, T const& value)
	{
		lua_State* interpreter = lua_proxy_traits<ValueWrapper>::interpreter(table);

		// TODO: Exception safe?

		lua_proxy_traits<ValueWrapper>::unwrap(interpreter, table);
		detail::stack_pop pop(interpreter, 1);
		detail::push(interpreter, key);
		detail::push(interpreter, value);
		lua_settable(interpreter, -3);
	}

	template<class ValueWrapper, class K>
	inline object rawget(ValueWrapper const& table, K const& key)
	{
		lua_State* interpreter = lua_proxy_traits<ValueWrapper>::interpreter(
			table
		);

		lua_proxy_traits<ValueWrapper>::unwrap(interpreter, table);
		detail::stack_pop pop(interpreter, 2);
		detail::push(interpreter, key);
		lua_rawget(interpreter, -2);
		return object(from_stack(interpreter, -1));
	}

	template<class ValueWrapper, class K, class T>
	inline void rawset(ValueWrapper const& table, K const& key, T const& value)
	{
		lua_State* interpreter = lua_proxy_traits<ValueWrapper>::interpreter(
			table
		);

		// TODO: Exception safe?

		lua_proxy_traits<ValueWrapper>::unwrap(interpreter, table);
		detail::stack_pop pop(interpreter, 1);
		detail::push(interpreter, key);
		detail::push(interpreter, value);
		lua_rawset(interpreter, -3);
	}

	template<class ValueWrapper>
	inline int type(ValueWrapper const& value)
	{
		lua_State* interpreter = lua_proxy_traits<ValueWrapper>::interpreter(value);

		lua_proxy_traits<ValueWrapper>::unwrap(interpreter, value);
		detail::stack_pop pop(interpreter, 1);
		return lua_type(interpreter, -1);
	}

	template <class ValueWrapper>
	inline object getmetatable(ValueWrapper const& obj)
	{
		lua_State* interpreter = lua_proxy_traits<ValueWrapper>::interpreter(obj);
		lua_proxy_traits<ValueWrapper>::unwrap(interpreter, obj);
		detail::stack_pop pop(interpreter, 2);
		lua_getmetatable(interpreter, -1);
		return object(from_stack(interpreter, -1));
	}

	template <class ValueWrapper1, class ValueWrapper2>
	inline void setmetatable(ValueWrapper1 const& obj, ValueWrapper2 const& metatable)
	{
		lua_State* interpreter = lua_proxy_traits<ValueWrapper1>::interpreter(obj);
		lua_proxy_traits<ValueWrapper1>::unwrap(interpreter, obj);
		detail::stack_pop pop(interpreter, 1);
		lua_proxy_traits<ValueWrapper2>::unwrap(interpreter, metatable);
		lua_setmetatable(interpreter, -2);
	}

	template <class ValueWrapper>
	inline std::tuple<const char*, object> getupvalue(ValueWrapper const& value, int index)
	{
		lua_State* interpreter = lua_proxy_traits<ValueWrapper>::interpreter(value);
		lua_proxy_traits<ValueWrapper>::unwrap(interpreter, value);
		detail::stack_pop pop(interpreter, 2);
		const char* name = lua_getupvalue(interpreter, -1, index);
		return std::make_tuple(name, object(from_stack(interpreter, -1)));
	}

	template <class ValueWrapper1, class ValueWrapper2>
	inline void setupvalue(ValueWrapper1 const& function, int index, ValueWrapper2 const& value)
	{
		lua_State* interpreter = lua_proxy_traits<ValueWrapper1>::interpreter(function);

		lua_proxy_traits<ValueWrapper1>::unwrap(interpreter, function);
		detail::stack_pop pop(interpreter, 1);
		lua_proxy_traits<ValueWrapper2>::unwrap(interpreter, value);
		lua_setupvalue(interpreter, -2, index);
	}

	template <class GetValueWrapper>
	object property(GetValueWrapper const& get)
	{
		lua_State* interpreter = lua_proxy_traits<GetValueWrapper>::interpreter(get);
		lua_proxy_traits<GetValueWrapper>::unwrap(interpreter, get);
		lua_pushnil(interpreter);
		lua_pushcclosure(interpreter, &detail::property_tag, 2);
		detail::stack_pop pop(interpreter, 1);
		return object(from_stack(interpreter, -1));
	}

	template <class GetValueWrapper, class SetValueWrapper>
	object property(GetValueWrapper const& get, SetValueWrapper const& set)
	{
		lua_State* interpreter = lua_proxy_traits<GetValueWrapper>::interpreter(get);
		lua_proxy_traits<GetValueWrapper>::unwrap(interpreter, get);
		lua_proxy_traits<SetValueWrapper>::unwrap(interpreter, set);
		lua_pushcclosure(interpreter, &detail::property_tag, 2);
		detail::stack_pop pop(interpreter, 1);
		return object(from_stack(interpreter, -1));
	}

} // namespace luabind

#if LUA_VERSION_NUM < 502
#undef lua_pushglobaltable
#endif

#endif // LUABIND_OBJECT_050419_HPP

