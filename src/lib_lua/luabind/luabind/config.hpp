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


#ifndef LUABIND_CONFIG_HPP_INCLUDED
#define LUABIND_CONFIG_HPP_INCLUDED

#include <boost/config.hpp>
#include <boost/version.hpp>
#include <boost/detail/workaround.hpp>

#include <luabind/build_information.hpp>

#ifdef BOOST_MSVC
    #define LUABIND_ANONYMOUS_FIX static
#else
    #define LUABIND_ANONYMOUS_FIX
#endif

#if BOOST_WORKAROUND(BOOST_MSVC, <= 1300)
    #error "Support for your version of Visual C++ has been removed from this version of Luabind"
#endif

// the maximum number of arguments of functions that's
// registered. Must at least be 2
#ifndef LUABIND_MAX_ARITY
    #define LUABIND_MAX_ARITY 10
#elif LUABIND_MAX_ARITY <= 1
    #undef LUABIND_MAX_ARITY
    #define LUABIND_MAX_ARITY 2
#endif

// the maximum number of classes one class
// can derive from
// max bases must at least be 1
#ifndef LUABIND_MAX_BASES
    #define LUABIND_MAX_BASES 4
#elif LUABIND_MAX_BASES <= 0
    #undef LUABIND_MAX_BASES
    #define LUABIND_MAX_BASES 1
#endif

// LUABIND_NO_ERROR_CHECKING
// define this to remove all error checks
// this will improve performance and memory
// footprint.
// if it is defined matchers will only be called on
// overloaded functions, functions that's
// not overloaded will be called directly. The
// parameters on the lua stack are assumed
// to match those of the function.
// exceptions will still be catched when there's
// no error checking.

// LUABIND_NOT_THREADSAFE
// this define will make luabind non-thread safe. That is,
// it will rely on a static variable. You can still have
// multiple lua states and use coroutines, but only
// one of your real threads may run lua code.

// LUABIND_NO_EXCEPTIONS
// this define will disable all usage of try, catch and throw in
// luabind. This will in many cases disable runtime-errors, such
// as invalid casts, when calling lua-functions that fails or
// returns values that cannot be converted by the given policy.
// Luabind requires that no function called directly or indirectly
// by luabind throws an exception (throwing exceptions through
// C code has undefined behavior, lua is written in C).

#ifdef LUABIND_DYNAMIC_LINK
# if defined (BOOST_WINDOWS)
#  ifdef LUABIND_BUILDING
#   define LUABIND_API __declspec(dllexport)
#  else
#   define LUABIND_API __declspec(dllimport)
#  endif
# elif defined (__CYGWIN__)
#  ifdef LUABIND_BUILDING
#   define LUABIND_API __attribute__ ((dllexport))
#  else
#   define LUABIND_API __attribute__ ((dllimport))
#  endif
# else
#  if defined(__GNUC__) && __GNUC__ >=4
#   define LUABIND_API __attribute__ ((visibility("default")))
#  endif
# endif
#endif

#ifndef LUABIND_API
# define LUABIND_API
#endif

// C++11 features //

#if (   defined(BOOST_NO_CXX11_SCOPED_ENUMS) \
     || defined(BOOST_NO_SCOPED_ENUMS)                 \
     || BOOST_VERSION < 105600                         \
        && (   defined(BOOST_NO_CXX11_HDR_TYPE_TRAITS) \
            || defined(BOOST_NO_0X_HDR_TYPE_TRAITS)))
#    define LUABIND_NO_SCOPED_ENUM
#endif

#if (   defined(BOOST_NO_CXX11_RVALUE_REFERENCES) \
     || defined(BOOST_NO_RVALUE_REFERENCES))
#    define LUABIND_NO_RVALUE_REFERENCES
#endif

// If you use Boost <= 1.46 but your compiler is C++11 compliant and marks
// destructors as noexcept by default, you need to define LUABIND_USE_NOEXCEPT.
#if (   !defined(BOOST_NO_NOEXCEPT) \
     && !defined(BOOST_NO_CXX11_NOEXCEPT) \
     && BOOST_VERSION >= 104700)
#    define LUABIND_USE_NOEXCEPT
#endif

#ifndef LUABIND_MAY_THROW
#    ifdef BOOST_NOEXCEPT_IF
#        define LUABIND_MAY_THROW BOOST_NOEXCEPT_IF(false)
#    elif defined(LUABIND_USE_NOEXCEPT)
#        define LUABIND_MAY_THROW noexcept(false)
#    else
#       define LUABIND_MAY_THROW
#    endif
#endif

#ifndef LUABIND_NOEXCEPT
#    ifdef BOOST_NOEXCEPT_OR_NOTHROW
#        define LUABIND_NOEXCEPT BOOST_NOEXCEPT_OR_NOTHROW
#    elif defined(LUABIND_USE_NOEXCEPT)
#        define LUABIND_NOEXCEPT noexcept
#    else
#       define LUABIND_NOEXCEPT throw()
#    endif
#endif

namespace luabind {

LUABIND_API void disable_super_deprecation();

} // namespace luabind

#endif // LUABIND_CONFIG_HPP_INCLUDED
