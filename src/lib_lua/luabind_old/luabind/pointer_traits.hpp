// Copyright (c) 2005 Daniel Wallin

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

#ifndef LUABIND_GET_POINTER_051023_HPP
# define LUABIND_GET_POINTER_051023_HPP
#include <memory>
#include <stdexcept>

// TODO: Rename to pointer_traits

namespace luabind {

	template<typename T>
	T* get_pointer(T* pointer)
	{
		return pointer;
	}

	template<typename T>
	T* get_pointer(const std::unique_ptr<T>& pointer)
	{
		return pointer.get();
	}

	template<typename T>
	T* get_pointer(const std::shared_ptr<T>& pointer)
	{
		return pointer.get();
	}


	namespace detail {

		template<typename T>
		struct pointer_traits {
			enum { is_pointer = false };
		};

		template<typename T>
		struct pointer_traits<T*>
		{
			enum { is_pointer = true };
			using value_type = T;
		};

		template<typename T>
		struct pointer_traits<std::unique_ptr<T>>
		{
			enum { is_pointer = true };
			using value_type = T;
		};

		template<typename T>
		struct pointer_traits<std::shared_ptr<T>>
		{
			enum { is_pointer = true };
			using value_type = T;
		};

		template<typename T>
		using is_pointer_to_const = std::is_const< typename pointer_traits<T>::value_type >;

		template<typename T>
		void release_ownership(std::unique_ptr<T>& p)
		{
			p.release();
		}

		template <class P>
		void release_ownership(P const&)
		{
			throw std::runtime_error(
				"luabind: smart pointer does not allow ownership transfer");
		}

		namespace has_get_pointer_
		{

			struct any
			{
				template<class T> any(T const&);
			};

			struct no_overload_tag
			{};

			typedef char(&yes)[1];
			typedef char(&no)[2];

			no_overload_tag operator, (no_overload_tag, int);

			template<class T>
			T* get_pointer(T const volatile*);

			template<class T>
			T* get_pointer(std::unique_ptr<T> const&);

			template<class T>
			T* get_pointer(std::shared_ptr<T> const&);

			detail::has_get_pointer_::no_overload_tag
				get_pointer(detail::has_get_pointer_::any);

			///@TODO: Rework
			template<class T>
			yes check(T const&);
			no check(no_overload_tag);

			template<class T>
			struct impl
			{
				static typename std::add_lvalue_reference<T>::type x;
				static const bool value = (sizeof(has_get_pointer_::check((get_pointer(x), 0))) == 1);
				typedef std::integral_constant<bool, value> type;
			};

		} // namespace has_get_pointer_

		template<class T>
		struct has_get_pointer
			: has_get_pointer_::impl<T>::type
		{};

	} // namespace detail

} // namespace luabind

#endif // LUABIND_GET_POINTER_051023_HPP

