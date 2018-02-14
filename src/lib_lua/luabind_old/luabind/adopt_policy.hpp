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


#ifndef LUABIND_ADOPT_POLICY_HPP_INCLUDED
#define LUABIND_ADOPT_POLICY_HPP_INCLUDED

#include <luabind/config.hpp>

#ifndef LUABIND_WRAPPER_BASE_HPP_INCLUDED
# include <luabind/wrapper_base.hpp>
#endif

#include <luabind/detail/policy.hpp>
#include <luabind/back_reference_fwd.hpp>

namespace luabind {
	namespace detail {

		template <class T>
		void adjust_backref_ownership(T* ptr, std::true_type)
		{
			if(wrap_base* p = dynamic_cast<wrap_base*>(ptr))
			{
				wrapped_self_t& wrapper = wrap_access::ref(*p);
				wrapper.get(wrapper.state());
				wrapper.m_strong_ref.set(wrapper.state());
			}
		}

		inline void adjust_backref_ownership(void*, std::false_type)
		{}

		template <class Pointer, class Direction = lua_to_cpp>
		struct adopt_pointer : pointer_converter
		{
			using type = adopt_pointer;

			enum { consumed_args = 1 };

			template<class T>
			T* to_cpp(lua_State* L, by_pointer<T>, int index)
			{
				T* ptr = pointer_converter::to_cpp(
					L, decorated_type<T*>(), index);

				object_rep* obj = static_cast<object_rep*>(
					lua_touserdata(L, index));
				obj->release();

				adjust_backref_ownership(ptr, std::is_polymorphic<T>());

				return ptr;
			}

			template<class T>
			int match(lua_State* L, by_pointer<T>, int index)
			{
				return pointer_converter::match(L, decorated_type<T*>(), index);
			}

			template<class T>
			void converter_postcall(lua_State*, T, int) {}
		};

		template <class Pointer, class T>
		struct pointer_or_default
		{
			using type = Pointer;
		};

		template <class T>
		struct pointer_or_default<void, T>
		{
			using type = std::unique_ptr<T>;
		};

		template <class Pointer>
		struct adopt_pointer<Pointer, cpp_to_lua>
		{
			using type = adopt_pointer;

			template<class T>
			void to_lua(lua_State* L, T* ptr)
			{
				if(ptr == 0)
				{
					lua_pushnil(L);
					return;
				}

				// if there is a back_reference, then the
				// ownership will be removed from the
				// back reference and put on the lua stack.
				if(luabind::move_back_reference(L, ptr))
					return;

				using pointer_type = typename pointer_or_default<Pointer, T>::type;

				make_pointer_instance(L, pointer_type(ptr));
			}
		};

		template <class Pointer>
		struct adopt_policy_impl
		{
			template<class T, class Direction>
			struct specialize
			{
				static_assert(detail::is_nonconst_pointer<T>::value, "Adopt policy only accepts non-const pointers");
				using type = adopt_pointer<Pointer, Direction>;
			};
		};

	}
}

namespace luabind
{
	// Caution: if we use the aliased type "policy_list" here, MSVC crashes.
	template<unsigned int N, typename Pointer = void>
	using adopt_policy = meta::type_list<converter_policy_injector<N, detail::adopt_policy_impl<Pointer>>>;
}

#endif // LUABIND_ADOPT_POLICY_HPP_INCLUDE

