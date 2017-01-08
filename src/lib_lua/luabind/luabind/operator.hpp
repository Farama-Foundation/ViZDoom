// Copyright (c) 2004 Daniel Wallin and Arvid Norberg

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

#ifndef OPERATOR_040729_HPP
#define OPERATOR_040729_HPP

#if defined(__GNUC__) && __GNUC__ < 3
# define LUABIND_NO_STRINGSTREAM
#else
# if defined(BOOST_NO_STRINGSTREAM)
#  define LUABIND_NO_STRINGSTREAM
# endif
#endif

#ifdef LUABIND_NO_STRINGSTREAM
#include <strstream>
#else
#include <sstream>
#endif

#include <luabind/detail/meta.hpp>
#include <luabind/lua_include.hpp>
#include <luabind/detail/other.hpp>
#include <luabind/detail/policy.hpp>

namespace luabind {
	namespace detail {

		template<class W, class T> struct unwrap_parameter_type;
		template<class Derived> struct operator_ {};

		struct operator_void_return {};

		template<class T>
		inline T const& operator,(T const& x, operator_void_return)
		{
			return x;
		}

		template<class Policies>
		inline void operator_result(lua_State*, operator_void_return, Policies*)
		{
		}

		template<class T, class Policies>
		inline void operator_result(lua_State* L, T const& x, Policies*)
		{
			specialized_converter_policy_n<0, Policies, T, cpp_to_lua >().to_lua(L, x);
		}

	}
} // namespace luabind


namespace luabind {
	namespace operators {

		template<class Self, typename... Args>
		struct call_operator
			: detail::operator_ < call_operator< Self, Args... > >
		{
			call_operator(int) {}

			template<class T, class Policies>
			struct apply
			{
				static void execute(
					lua_State* L
					, typename detail::unwrap_parameter_type<T, Self>::type self
					, typename detail::unwrap_parameter_type<T, Args>::type... args
				)
				{
					using namespace detail;
					operator_result(
						L
						, (self(args...), detail::operator_void_return())
						, (Policies*)0
					);
				}
			};

			static char const* name() { return "__call"; }
		};

	}
} // namespace luabind::operators

namespace luabind {

	template<class Derived>
	struct self_base
	{
		template< typename... Args >
		operators::call_operator<Derived, Args...> operator()(const Args&...) const
		{
			return 0;
		}
	};

	struct self_type : self_base<self_type>
	{
	};

	struct const_self_type : self_base<const_self_type>
	{
	};

	namespace detail {

		template<class W, class T>
		struct unwrap_parameter_type
		{
			using type = typename meta::select_ <
				meta::case_< std::is_same<T, self_type>, W& >,
				meta::case_< std::is_same<T, const_self_type >, W const& >,
				meta::default_< typename unwrap_other<T>::type >
			> ::type;
		};

		template<class Derived, class A, class B>
		struct binary_operator
			: operator_<binary_operator<Derived, A, B> >
		{
			binary_operator(int) {}

			template<class T, class Policies>
			struct apply
			{
				using arg0 = typename unwrap_parameter_type<T, A>::type;
				using arg1 = typename unwrap_parameter_type<T, B>::type;

				static void execute(lua_State* L, arg0 _0, arg1 _1)
				{
					Derived::template apply<arg0, arg1, Policies>::execute(
						L, _0, _1);
				}
			};

			static char const* name()
			{
				return Derived::name();
			}
		};

		template<class Derived, class A>
		struct unary_operator
			: operator_<unary_operator<Derived, A> >
		{
			unary_operator(int) {}

			template<class T, class Policies>
			struct apply
			{
				using arg0 = typename unwrap_parameter_type<T, A>::type;

				static void execute(lua_State* L, arg0 _0)
				{
					Derived::template apply<arg0, Policies>::execute(L, _0);
				}
			};

			static char const* name()
			{
				return Derived::name();
			}
		};

	}
} // namespace detail::luabind

namespace luabind {

#define LUABIND_BINARY_OPERATOR(name_, op) \
    namespace operators { \
\
        struct name_ \
        { \
            template<class T0, class T1, class Policies> \
            struct apply \
            { \
                static void execute(lua_State* L, T0 _0, T1 _1) \
                { \
                    detail::operator_result(L, _0 op _1, (Policies*)0); \
                } \
            }; \
\
            static char const* name() \
            { \
                return "__" # name_; \
            } \
        }; \
\
    } \
    \
    template<class T, class U> \
    detail::binary_operator< \
        operators::name_ \
      , U \
      , T \
    > \
    inline operator op(self_base<U>, T const&) \
    { \
        return 0; \
    } \
    \
    template<class T, class U> \
    detail::binary_operator< \
        operators::name_ \
      , T \
      , U \
    > \
    inline operator op(T const&, self_base<U>) \
    { \
        return 0; \
    } \
    \
    detail::binary_operator< \
        operators::name_ \
      , self_type \
      , self_type \
    > \
    inline operator op(self_type, self_type) \
    { \
        return 0; \
    } \
    \
    detail::binary_operator< \
        operators::name_ \
      , self_type \
      , const_self_type \
    > \
    inline operator op(self_type, const_self_type) \
    { \
        return 0; \
    } \
    \
    detail::binary_operator< \
        operators::name_ \
      , const_self_type \
      , self_type \
    > \
    inline operator op(const_self_type, self_type) \
    { \
        return 0; \
    } \
    \
    detail::binary_operator< \
        operators::name_ \
      , const_self_type \
      , const_self_type \
    > \
    inline operator op(const_self_type, const_self_type) \
    { \
        return 0; \
    }

	LUABIND_BINARY_OPERATOR(add, +)
		LUABIND_BINARY_OPERATOR(sub, -)
		LUABIND_BINARY_OPERATOR(mul, *)
		LUABIND_BINARY_OPERATOR(div, / )
		LUABIND_BINARY_OPERATOR(mod, %)
		LUABIND_BINARY_OPERATOR(pow, ^)
		LUABIND_BINARY_OPERATOR(lt, < )
		LUABIND_BINARY_OPERATOR(le, <= )
		LUABIND_BINARY_OPERATOR(eq, == )

#undef LUABIND_BINARY_OPERATOR

#define LUABIND_UNARY_OPERATOR(name_, op, fn) \
    namespace operators { \
\
        struct name_ \
        { \
            template<class T, class Policies> \
            struct apply \
            { \
                static void execute(lua_State* L, T x) \
                { \
                    detail::operator_result(L, op(x), (Policies*)0); \
                } \
            }; \
\
            static char const* name() \
            { \
                return "__" # name_; \
            } \
        }; \
\
    } \
    \
    template<class T> \
    detail::unary_operator< \
        operators::name_ \
      , T \
    > \
    inline fn(self_base<T>) \
    { \
        return 0; \
    }

		template<class T>
	std::string tostring_operator(T const& x)
	{
#ifdef LUABIND_NO_STRINGSTREAM
		std::strstream s;
		s << x << std::ends;
#else
		std::stringstream s;
		s << x;
#endif
		return s.str();
	}

	LUABIND_UNARY_OPERATOR(tostring, tostring_operator, tostring)
		LUABIND_UNARY_OPERATOR(unm, -, operator-)

#undef LUABIND_UNARY_OPERATOR


	extern LUABIND_API self_type self;
	extern LUABIND_API const_self_type const_self;


} // namespace luabind

#endif // OPERATOR_040729_HPP

