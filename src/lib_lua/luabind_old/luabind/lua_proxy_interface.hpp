#ifndef LUA_PROXY_INTERFACE_HPP_INCLUDED
#define LUA_PROXY_INTERFACE_HPP_INCLUDED

#include <luabind/lua_proxy.hpp>
#include <luabind/detail/call_function.hpp>
#include <ostream>

#if LUA_VERSION_NUM < 502
# define lua_compare(L, index1, index2, fn) fn(L, index1, index2)
# define LUA_OPEQ lua_equal
# define LUA_OPLT lua_lessthan
# define lua_rawlen lua_objlen
# define lua_pushglobaltable(L) lua_pushvalue(L, LUA_GLOBALSINDEX)
#endif

namespace luabind {

	namespace detail
	{

		template<class T, typename... Policies>
		void push(lua_State* interpreter, T& value, policy_list<Policies...> const& = no_policies())
		{
			using PolicyList = policy_list<Policies...>;
			using unwrapped_type = T;
			using converter_type = specialized_converter_policy_n<0, PolicyList, T, cpp_to_lua >;
			converter_type().to_lua(interpreter, implicit_cast<unwrapped_type&>(value));
		}

	} // namespace detail

	namespace adl {

		template <class T>
		class lua_proxy_interface;

		///@TODO: replace by decltype construct
		namespace is_object_interface_aux
		{
			typedef char(&yes)[1];
			typedef char(&no)[2];

			template <class T>
			yes check(lua_proxy_interface<T>*);
			no check(void*);

			template <class T>
			struct impl
			{
				static const bool value = sizeof(is_object_interface_aux::check((T*)0)) == sizeof(yes);
				typedef std::integral_constant<bool, value> type;
			};

		} // namespace is_object_interface_aux

		template <class T>
		struct is_object_interface :
			is_object_interface_aux::impl<T>::type
		{};

		template <class R, class T, class U>
		struct enable_binary
			: std::enable_if< is_object_interface<T>::value || is_object_interface<U>::value, R >
		{};

		template<class T, class U>
		int binary_interpreter(lua_State*& L, T const& lhs, U const& rhs, std::true_type, std::true_type)
		{
			L = lua_proxy_traits<T>::interpreter(lhs);
			lua_State* L2 = lua_proxy_traits<U>::interpreter(rhs);

			// you are comparing objects with different interpreters
			// that's not allowed.
			assert(L == L2 || L == 0 || L2 == 0);

			// if the two objects we compare have different interpreters
			// then they

			if(L != L2) return -1;
			if(L == 0) return 1;
			return 0;
		}

		template<class T, class U>
		int binary_interpreter(lua_State*& L, T const& x, U const&, std::true_type, std::false_type)
		{
			L = lua_proxy_traits<T>::interpreter(x);
			return 0;
		}

		template<class T, class U>
		int binary_interpreter(lua_State*& L, T const&, U const& x, std::false_type, std::true_type)
		{
			L = lua_proxy_traits<U>::interpreter(x);
			return 0;
		}

		template<class T, class U>
		int binary_interpreter(lua_State*& L, T const& x, U const& y)
		{
			return binary_interpreter(L, x, y, is_lua_proxy_type<T>(), is_lua_proxy_type<U>());
		}

		template<class LHS, class RHS>
		typename enable_binary<bool, LHS, RHS>::type
			operator==(LHS const& lhs, RHS const& rhs)
		{
			lua_State* L = 0;
			switch(binary_interpreter(L, lhs, rhs)) {
			case  1: return true;
			case-1: return false;
			}
			assert(L);
			detail::stack_pop pop1(L, 1);
			detail::push(L, lhs);
			detail::stack_pop pop2(L, 1);
			detail::push(L, rhs);
			return lua_compare(L, -1, -2, LUA_OPEQ) != 0;
		}

		template<class LHS, class RHS>
		typename enable_binary<bool, LHS, RHS>::type
			operator<(LHS const& lhs, RHS const& rhs)
		{
			lua_State* L = 0;
			switch(binary_interpreter(L, lhs, rhs)) {
			case  1: return true;
			case-1: return false;
			}
			assert(L);
			detail::stack_pop pop1(L, 1);
			detail::push(L, lhs);
			detail::stack_pop pop2(L, 1);
			detail::push(L, rhs);
			return lua_compare(L, -1, -2, LUA_OPLT) != 0;
		}

		template<class ValueWrapper>
		std::ostream& operator<<(std::ostream& os, lua_proxy_interface<ValueWrapper> const& v)
		{
			using namespace luabind;
			lua_State* interpreter = lua_proxy_traits<ValueWrapper>::interpreter(
				static_cast<ValueWrapper const&>(v));
			detail::stack_pop pop(interpreter, 1);
			lua_proxy_traits<ValueWrapper>::unwrap(interpreter, static_cast<ValueWrapper const&>(v));
			char const* p = lua_tostring(interpreter, -1);
			std::size_t len = lua_rawlen(interpreter, -1);
			os.write(p, len);
			//std::copy(p, p+len, std::ostream_iterator<char>(os));
			return os;
		}


		template<class LHS, class RHS>
		typename enable_binary<bool, LHS, RHS>::type
			operator>(LHS const& lhs, RHS const& rhs)
		{
			return !(lhs < rhs || lhs == rhs);
		}

		template<class LHS, class RHS>
		typename enable_binary<bool, LHS, RHS>::type
			operator<=(LHS const& lhs, RHS const& rhs)
		{
			return lhs < rhs || lhs == rhs;
		}

		template<class LHS, class RHS>
		typename enable_binary<bool, LHS, RHS>::type
			operator>=(LHS const& lhs, RHS const& rhs)
		{
			return !(lhs < rhs);
		}

		template<class LHS, class RHS>
		typename enable_binary<bool, LHS, RHS>::type
			operator!=(LHS const& lhs, RHS const& rhs)
		{
			return !(lhs == rhs);
		}

		template<class Derived>
		class lua_proxy_interface
		{
		public:
			~lua_proxy_interface() {}

			// defined in luabind/detail/object.hpp
			template<typename... Args>
			object operator()(Args&&... args);

			// defined in luabind/detail/object.hpp
			template<typename PolicyList, typename... Args>
			object call(Args&&... args);

			explicit operator bool() const
			{
				lua_State* L = lua_proxy_traits<Derived>::interpreter(derived());
				if(!L) return 0;
				lua_proxy_traits<Derived>::unwrap(L, derived());
				detail::stack_pop pop(L, 1);
				return lua_toboolean(L, -1) == 1;
			}

		private:
			Derived& derived() { return *static_cast<Derived*>(this); }
			Derived const& derived() const { return *static_cast<Derived const*>(this); }
		};

	}

	template<class ValueWrapper>
	std::string to_string(adl::lua_proxy_interface<ValueWrapper> const& v)
	{
		using namespace luabind;
		lua_State* interpreter = lua_proxy_traits<ValueWrapper>::interpreter(static_cast<ValueWrapper const&>(v));
		detail::stack_pop pop(interpreter, 1);
		lua_proxy_traits<ValueWrapper>::unwrap(interpreter, static_cast<ValueWrapper const&>(v));
		char const* p = lua_tostring(interpreter, -1);
		std::size_t len = lua_rawlen(interpreter, -1);
		return std::string(p, len);
	}

	namespace detail
	{
		template<class T, class ValueWrapper, class Policies, class ErrorPolicy, class ReturnType >
		ReturnType object_cast_aux(ValueWrapper const& value_wrapper, T*, Policies*, ErrorPolicy error_policy, ReturnType*)
		{
			lua_State* interpreter = lua_proxy_traits<ValueWrapper>::interpreter(value_wrapper);

#ifndef LUABIND_NO_ERROR_CHECKING
			if(!interpreter)
				return error_policy.handle_error(interpreter, typeid(void));
#endif
			lua_proxy_traits<ValueWrapper>::unwrap(interpreter, value_wrapper);
			detail::stack_pop pop(interpreter, 1);
			specialized_converter_policy_n<0, Policies, T, lua_to_cpp> cv;

			if(cv.match(interpreter, decorated_type<T>(), -1) < 0) {
				return error_policy.handle_error(interpreter, typeid(T));
			}
			return cv.to_cpp(interpreter, decorated_type<T>(), -1);
		}

		template<class T>
		struct throw_error_policy
		{
			T handle_error(lua_State* interpreter, type_id const& type_info)
			{
#ifndef LUABIND_NO_EXCEPTIONS
				throw cast_failed(interpreter, type_info);
#else
				cast_failed_callback_fun e = get_cast_failed_callback();
				if(e) e(interpreter, type_info);

				assert(0 && "object_cast failed. If you want to handle this error use "
					"luabind::set_error_callback()");
				std::terminate();
#endif
				return *(typename std::remove_reference<T>::type*)0;
			}
		};

		template<class T>
		struct nothrow_error_policy
		{
			nothrow_error_policy(T rhs) : value(rhs) {}

			T handle_error(lua_State*, type_id const&)
			{
				return value;
			}
		private:
			T value;
		};
	} // namespace detail

	template<class T, class ValueWrapper> inline
		T object_cast(ValueWrapper const& value_wrapper)
	{
		return detail::object_cast_aux(value_wrapper, (T*)0, (no_policies*)0, detail::throw_error_policy<T>(), (T*)0);
	}

	template<class T, class ValueWrapper, class Policies> inline
		T object_cast(ValueWrapper const& value_wrapper, Policies const&)
	{
		return detail::object_cast_aux(value_wrapper, (T*)0, (Policies*)0, detail::throw_error_policy<T>(), (T*)0);
	}

	template<typename T, typename ValueWrapper, typename ReturnValue> inline
		ReturnValue object_cast_nothrow(ValueWrapper const& value_wrapper, ReturnValue default_value)
	{
		return detail::object_cast_aux(value_wrapper, (T*)0, (no_policies*)0, detail::nothrow_error_policy<ReturnValue>(default_value), (ReturnValue*)0);
	}

	template<typename T, typename ValueWrapper, typename Policies, typename ReturnValue> inline
		ReturnValue object_cast_nothrow(ValueWrapper const& value_wrapper, Policies const&, ReturnValue default_value)
	{
		return detail::object_cast_aux(value_wrapper, (T*)0, (Policies*)0, detail::nothrow_error_policy<ReturnValue>(default_value), (ReturnValue*)0);
	}

	template <class ValueWrapper>
	inline lua_CFunction tocfunction(ValueWrapper const& value)
	{
		lua_State* interpreter = lua_proxy_traits<ValueWrapper>::interpreter(value);
		lua_proxy_traits<ValueWrapper>::unwrap(interpreter, value);
		detail::stack_pop pop(interpreter, 1);
		return lua_tocfunction(interpreter, -1);
	}

	template <class T, class ValueWrapper>
	inline T* touserdata(ValueWrapper const& value)
	{
		lua_State* interpreter = lua_proxy_traits<ValueWrapper>::interpreter(value);

		lua_proxy_traits<ValueWrapper>::unwrap(interpreter, value);
		detail::stack_pop pop(interpreter, 1);
		return static_cast<T*>(lua_touserdata(interpreter, -1));
	}

}

#if LUA_VERSION_NUM < 502
# undef lua_compare
# undef LUA_OPEQ
# undef LUA_OPLT
# undef lua_rawlen
# undef lua_pushglobaltable
#endif

#endif

