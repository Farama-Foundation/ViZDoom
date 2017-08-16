#ifndef LUA_ITERATOR_PROXY_HPP_INCLUDED
#define LUA_ITERATOR_PROXY_HPP_INCLUDED

#include <luabind/lua_proxy_interface.hpp>
#include <luabind/lua_index_proxy.hpp>
#include <luabind/handle.hpp>
#include <luabind/nil.hpp>
#include <luabind/lua_include.hpp>
#include <luabind/detail/crtp_iterator.hpp>

#if LUA_VERSION_NUM < 502
# define lua_compare(L, index1, index2, fn) fn(L, index1, index2)
# define LUA_OPEQ lua_equal
#endif

namespace luabind {

	namespace adl {

		template<class AccessPolicy>
		class iterator_proxy :
			public lua_proxy_interface<iterator_proxy<AccessPolicy> >
		{
		public:
			iterator_proxy(lua_State* interpreter, handle const& table, handle const& key)
				: m_interpreter(interpreter), m_table_index(lua_gettop(interpreter) + 1), m_key_index(m_table_index + 1)
			{
				table.push(m_interpreter);
				key.push(m_interpreter);
			}

			iterator_proxy(iterator_proxy const& other)
				: m_interpreter(other.m_interpreter), m_table_index(other.m_table_index), m_key_index(other.m_key_index)
			{
				other.m_interpreter = 0;
			}

			~iterator_proxy()
			{
				if(m_interpreter) lua_pop(m_interpreter, 2);
			}

			// this will set the value to nil
			iterator_proxy & operator=(luabind::detail::nil_type)
			{
				lua_pushvalue(m_interpreter, m_key_index);
				lua_pushnil(m_interpreter);
				AccessPolicy::set(m_interpreter, m_table_index);
				return *this;
			}

			template<class T>
			iterator_proxy& operator=(T const& value)
			{
				lua_pushvalue(m_interpreter, m_key_index);
				detail::push(m_interpreter, value);
				AccessPolicy::set(m_interpreter, m_table_index);
				return *this;
			}

			template<class Key>
			index_proxy<iterator_proxy<AccessPolicy> > operator[](Key const& key)
			{
				return index_proxy<iterator_proxy<AccessPolicy> >(
					*this, m_interpreter, key
					);
			}

			// This is non-const to prevent conversion on lvalues.
			// defined in luabind/object.hpp
			operator object();

			lua_State* interpreter() const
			{
				return m_interpreter;
			}

			// TODO: Why is it non-const?
			void push(lua_State* interpreter)
			{
				assert(interpreter == m_interpreter);
				lua_pushvalue(m_interpreter, m_key_index);
				AccessPolicy::get(m_interpreter, m_table_index);
			}

		private:
			mutable lua_State* m_interpreter;
			int m_table_index;
			int m_key_index;
		};

	}

	template<class AccessPolicy>
	struct lua_proxy_traits<adl::iterator_proxy<AccessPolicy> >
	{
		using is_specialized = std::true_type;

		template<class Proxy>
		static lua_State* interpreter(Proxy const& p)
		{
			return p.interpreter();
		}

		template<class Proxy>
		static void unwrap(lua_State* interpreter, Proxy const& p)
		{
			// TODO: Why const_cast?
			const_cast<Proxy&>(p).push(interpreter);
		}
	};

	namespace detail
	{
		struct basic_access
		{
			static void set(lua_State* interpreter, int table)
			{
				lua_settable(interpreter, table);
			}

			static void get(lua_State* interpreter, int table)
			{
				lua_gettable(interpreter, table);
			}
		};

		struct raw_access
		{
			static void set(lua_State* interpreter, int table)
			{
				lua_rawset(interpreter, table);
			}

			static void get(lua_State* interpreter, int table)
			{
				lua_rawget(interpreter, table);
			}
		};

		template<class AccessPolicy>
		class basic_iterator :
			public detail::crtp_iterator< basic_iterator<AccessPolicy>, adl::iterator_proxy<AccessPolicy>, std::forward_iterator_tag, adl::iterator_proxy<AccessPolicy> >
		{
		public:
			basic_iterator()
				: m_interpreter(0)
			{}

			template<class ValueWrapper>
			explicit basic_iterator(ValueWrapper const& value_wrapper)
				: m_interpreter(lua_proxy_traits<ValueWrapper>::interpreter(value_wrapper))
			{
				detail::stack_pop pop(m_interpreter, 1);
				lua_proxy_traits<ValueWrapper>::unwrap(m_interpreter, value_wrapper);

				lua_pushnil(m_interpreter);
				if(lua_next(m_interpreter, -2) != 0) {
					detail::stack_pop pop(m_interpreter, 2);
					handle(m_interpreter, -2).swap(m_key);
				}
				else {
					m_interpreter = 0;
					return;
				}

				handle(m_interpreter, -1).swap(m_table);
			}

			// defined in luabind/detail/object.hpp
			adl::object key() const;

		private:
			template< typename, typename, typename, typename, typename >
			friend class detail::crtp_iterator;

			void increment()
			{
				m_table.push(m_interpreter);
				m_key.push(m_interpreter);

				detail::stack_pop pop(m_interpreter, 1);

				if(lua_next(m_interpreter, -2) != 0) {
					m_key.replace(m_interpreter, -2);
					lua_pop(m_interpreter, 2);
				}
				else {
					m_interpreter = 0;
					handle().swap(m_table);
					handle().swap(m_key);
				}
			}

			bool equal(basic_iterator const& other) const
			{
				if(m_interpreter == 0 && other.m_interpreter == 0)
					return true;

				if(m_interpreter != other.m_interpreter)
					return false;

				detail::stack_pop pop(m_interpreter, 2);
				m_key.push(m_interpreter);
				other.m_key.push(m_interpreter);
				return lua_compare(m_interpreter, -2, -1, LUA_OPEQ) != 0;
			}

			adl::iterator_proxy<AccessPolicy> dereference() const
			{
				return adl::iterator_proxy<AccessPolicy>(m_interpreter, m_table, m_key);
			}

			lua_State* m_interpreter;
			handle m_table;
			handle m_key;
		};

	} // namespace detail

	using iterator     = detail::basic_iterator<detail::basic_access>;
	using raw_iterator = detail::basic_iterator<detail::raw_access>;

}

#if LUA_VERSION_NUM < 502
#undef LUA_OPEQ
#undef lua_compare
#endif

#endif

