// Copyright Daniel Wallin 2009. Use, modification and distribution is
// subject to the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef LUABIND_SHARED_PTR_CONVERTER_090211_HPP
# define LUABIND_SHARED_PTR_CONVERTER_090211_HPP

#include <luabind/detail/policy.hpp>
#include <luabind/detail/conversion_policies/value_converter.hpp>    // for default_converter, etc
#include <luabind/get_main_thread.hpp>  // for get_main_thread
#include <luabind/handle.hpp>           // for handle
#include <luabind/detail/decorate_type.hpp>  // for decorated_type
#include <memory>

namespace luabind {

	namespace detail
	{

		struct shared_ptr_deleter
		{
			shared_ptr_deleter(lua_State* L, int index)
				: life_support(get_main_thread(L), L, index)
			{}

			void operator()(void const*)
			{
				handle().swap(life_support);
			}

			handle life_support;
		};

	} // namespace detail

	template <class T>
	struct default_converter<std::shared_ptr<T> >
		: default_converter<T*>
	{
		using is_native = std::false_type;

		template <class U>
		int match(lua_State* L, U, int index)
		{
			return default_converter<T*>::match(
				L, decorated_type<T*>(), index);
		}

		template <class U>
		std::shared_ptr<T> to_cpp(lua_State* L, U, int index)
		{
			T* raw_ptr = default_converter<T*>::to_cpp(
				L, decorated_type<T*>(), index);
			if(!raw_ptr)
				return std::shared_ptr<T>();
			return std::shared_ptr<T>(
				raw_ptr, detail::shared_ptr_deleter(L, index));
		}

		void to_lua(lua_State* L, std::shared_ptr<T> const& p)
		{
			if(detail::shared_ptr_deleter* d =
				std::get_deleter<detail::shared_ptr_deleter>(p))
			{
				d->life_support.push(L);
			}
			else
			{
				detail::value_converter().to_lua(L, p);
			}
		}

		template <class U>
		void converter_postcall(lua_State*, U const&, int)
		{}
	};

	template <class T>
	struct default_converter<std::shared_ptr<T> const&>
		: default_converter<std::shared_ptr<T> >
	{};

} // namespace luabind

#endif // LUABIND_SHARED_PTR_CONVERTER_090211_HPP

