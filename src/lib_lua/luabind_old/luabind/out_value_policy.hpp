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


#ifndef LUABIND_OUT_VALUE_POLICY_HPP_INCLUDED
#define LUABIND_OUT_VALUE_POLICY_HPP_INCLUDED

#include <luabind/config.hpp>
#include <luabind/detail/policy.hpp>    // for find_conversion_policy, etc
#include <luabind/detail/decorate_type.hpp>  // for decorated_type
#include <luabind/detail/primitives.hpp>  // for by_pointer, by_reference, etc
#include <luabind/detail/typetraits.hpp>  // for is_nonconst_pointer, is_nonconst_reference, etc
#include <new>                          // for operator new

namespace luabind {
	namespace detail {

		template<int N>
		struct char_array
		{
			char storage[N];
		};

		template<class U>
		char_array<sizeof(typename identity<U>::type)> indirect_sizeof_test(by_reference<U>);

		template<class U>
		char_array<sizeof(typename identity<U>::type)> indirect_sizeof_test(by_const_reference<U>);

		template<class U>
		char_array<sizeof(typename identity<U>::type)> indirect_sizeof_test(by_pointer<U>);

		template<class U>
		char_array<sizeof(typename identity<U>::type)> indirect_sizeof_test(by_const_pointer<U>);

		template<class U>
		char_array<sizeof(typename identity<U>::type)> indirect_sizeof_test(by_value<U>);

		template<class T>
		struct indirect_sizeof
		{
			static const int value = sizeof(indirect_sizeof_test(decorated_type<T>()));
		};

		namespace out_value_detail {

			template< int Size >
			struct temporary_storage_size {

				template< typename T, typename... Args >
				void construct(Args&&... args)
				{
					new (&m_storage) T(std::forward<Args>(args)...);
				}

				template<typename T>
				T& get() {
					return *reinterpret_cast<T*>(&m_storage);
				}

				template<typename T>
				const T& get() const {
					return *reinterpret_cast<T*>(&m_storage);
				}

				template<typename T>
				void destroy()
				{
					get<T>().~T();
				}

				typename std::aligned_storage<Size>::type m_storage;

			};

			template< typename T >
			struct temporary_storage_type {

				template< typename... Args >
				void construct(Args&&... args)
				{
					new (&m_storage) T(std::forward<Args>(args)...);
				}

				T& get() {
					return *reinterpret_cast<T*>(&m_storage);
				}

				const T& get() const {
					return *reinterpret_cast<T*>(&m_storage);
				}

				void destroy()
				{
					get().~T();
				}

				typename std::aligned_storage<sizeof(T), alignof(T)>::type m_storage;

			};

		}

		// See note in out_value_policy about why we're not templating
		// for the parameter type.
		template<typename T, class Policies = no_policies>
		struct out_value_converter
		{
			enum { consumed_args = 1 };

			T& to_cpp(lua_State* L, by_reference<T>, int index)
			{
				//specialized_converter_policy_n<1, Policies, T, lua_to_cpp> converter;
				storage_.construct(converter_.to_cpp(L, decorated_type<T>(), index));
				return storage_.get();
			}

			int match(lua_State* L, by_reference<T>, int index)
			{
				return converter_.match(L, decorated_type<T>(), index);
			}

			void converter_postcall(lua_State* L, by_reference<T>, int)
			{
				//specialized_converter_policy_n<2,Policies,T,cpp_to_lua> converter;
				converter_.to_lua(L, storage_.get());
				storage_.destroy();
			}

			T* to_cpp(lua_State* L, by_pointer<T>, int index)
			{
				storage_.construct(converter_.to_cpp(L, decorated_type<T>(), index));
				return &storage_.get();
			}

			int match(lua_State* L, by_pointer<T>, int index)
			{
				return converter_.match(L, decorated_type<T>(), index);
			}

			void converter_postcall(lua_State* L, by_pointer<T>, int)
			{
				//specialized_converter_policy_n<2, Policies, T, cpp_to_lua> converter;
				converter_.to_lua(L, storage_.get());
				storage_.destroy();
			}

		private:
			specialized_converter_policy_n<1, Policies, T, lua_to_cpp > converter_;
			out_value_detail::temporary_storage_type<T> storage_;
		};

		template<class Policies = no_policies>
		struct out_value_policy
		{
			struct only_accepts_nonconst_references_or_pointers {};
			struct can_only_convert_from_lua_to_cpp {};

			template<class T, class Direction>
			struct specialize
			{
				static_assert(std::is_same< Direction, lua_to_cpp >::value, "Out value policy can only convert from lua to cpp");
				static_assert(meta::or_< is_nonconst_reference<T>, is_nonconst_pointer<T> >::value, "Out value policy only accepts non const references or pointers");

				// Note to myself:
				// Using the size and template members instead of a policy templated for the type seems
				// to be done to tame template bloat. Need to check if this is worth is.
				using base_type = typename std::remove_pointer< typename std::remove_reference< T >::type >::type;
				using type = out_value_converter<base_type, Policies>;
			};
		};

		template<int Size, class Policies = no_policies>
		struct pure_out_value_converter
		{
			enum { consumed_args = 0 };

			template<class T>
			T& to_cpp(lua_State*, by_reference<T>, int)
			{
				storage_.template construct<T>();
				return storage_.template get<T>();
			}

			template<class T>
			static int match(lua_State*, by_reference<T>, int)
			{
				return 0;
			}

			template<class T>
			void converter_postcall(lua_State* L, by_reference<T>, int)
			{
				specialized_converter_policy_n<1, Policies, T, cpp_to_lua> converter;
				converter.to_lua(L, storage_.template get<T>());
				storage_.template destroy<T>();
			}

			template<class T>
			T* to_cpp(lua_State*, by_pointer<T>, int)
			{
				storage_.template construct<T>();
				return &storage_.template get<T>();
			}

			template<class T>
			static int match(lua_State*, by_pointer<T>, int)
			{
				return 0;
			}

			template<class T>
			void converter_postcall(lua_State* L, by_pointer<T>, int)
			{
				specialized_converter_policy_n<1, Policies, T, cpp_to_lua> converter;
				converter.to_lua(L, storage_.template get<T>());
				storage_.template destroy<T>();
			}

		private:
			out_value_detail::temporary_storage_size<Size> storage_;
		};

		template<class Policies = no_policies>
		struct pure_out_value_policy
		{
			struct only_accepts_nonconst_references_or_pointers {};
			struct can_only_convert_from_lua_to_cpp {};

			template<class T, class Direction>
			struct specialize
			{
				static_assert(std::is_same< Direction, lua_to_cpp >::value, "Pure out value policy can only convert from lua to cpp");
				static_assert(meta::or_< is_nonconst_reference<T>, is_nonconst_pointer<T> >::value, "Pure out value policy only accepts non const references or pointers");

				using type = pure_out_value_converter<indirect_sizeof<T>::value, Policies>;
			};
		};

	}
}

namespace luabind
{
	template<unsigned int N, class Policies = no_policies >
	using out_value = meta::type_list<converter_policy_injector<N, detail::out_value_policy<Policies>>>;

	template<unsigned int N, class Policies = no_policies >
	using pure_out_value = meta::type_list<converter_policy_injector<N, detail::pure_out_value_policy<Policies>>>;
}

#endif // LUABIND_OUT_VALUE_POLICY_HPP_INCLUDED

