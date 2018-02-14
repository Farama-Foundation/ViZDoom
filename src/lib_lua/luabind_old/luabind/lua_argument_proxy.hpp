#ifndef LUA_ARGUMENT_PROXY_HPP_INCLUDED
#define LUA_ARGUMENT_PROXY_HPP_INCLUDED

#include <luabind/lua_include.hpp>
#include <luabind/lua_proxy_interface.hpp>
#include <luabind/lua_index_proxy.hpp>
#include <luabind/from_stack.hpp>

namespace luabind {

	namespace adl {

		class argument : public lua_proxy_interface<argument>
		{
		public:
			argument(from_stack const& stack_reference)
				: m_interpreter(stack_reference.interpreter), m_index(stack_reference.index)
			{
				if(m_index < 0) m_index = lua_gettop(m_interpreter) - m_index + 1;
			}

			template<class T>
			index_proxy<argument> operator[](T const& key) const
			{
				return index_proxy<argument>(*this, m_interpreter, key);
			}

			void push(lua_State* L) const
			{
				lua_pushvalue(L, m_index);
			}

			lua_State* interpreter() const
			{
				return m_interpreter;
			}

		private:
			lua_State* m_interpreter;
			int m_index;
		};

	}

	using adl::argument;

	template<>
	struct lua_proxy_traits<argument>
	{
		using is_specialized = std::true_type;

		static lua_State* interpreter(argument const& value)
		{
			return value.interpreter();
		}

		static void unwrap(lua_State* interpreter, argument const& value)
		{
			value.push(interpreter);
		}

		static bool check(...)
		{
			return true;
		}
	};

}

#endif

