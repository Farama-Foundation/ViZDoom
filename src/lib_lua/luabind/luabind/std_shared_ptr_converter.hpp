// Copyright Christian Neumüller 2012. Use, modification and distribution is
// subject to the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef LUABIND_STD_SHAREDPTR_CONVERTER_HPP_INCLUDED
#define LUABIND_STD_SHAREDPTR_CONVERTER_HPP_INCLUDED LUABIND_STD_SHAREDPTR_CONVERTER_HPP_INCLUDED

#include <boost/config.hpp>
#if defined(LUABIND_NO_STD_SHARED_PTR)    \
    || defined(BOOST_NO_CXX11_SMART_PTR)  \
    && !defined(BOOST_HAS_TR1_SHARED_PTR) \
    && (!defined(BOOST_MSVC) || BOOST_MSVC < 1600)
#  ifndef LUABIND_NO_STD_SHARED_PTR
#    define LUABIND_NO_STD_SHARED_PTR
#  endif
#else

# include <luabind/shared_ptr_converter.hpp>
# include <memory> // shared_ptr

# if BOOST_VERSION >= 105300
#  include <luabind/detail/has_get_pointer.hpp>

#  include <boost/get_pointer.hpp>

namespace luabind { namespace detail { namespace has_get_pointer_ {
  template<class T>
  struct impl<std::shared_ptr<T>> {
      BOOST_STATIC_CONSTANT(bool, value = true);
      typedef boost::mpl::bool_<value> type;
  };

  template<class T>
  struct impl<const std::shared_ptr<T>>: impl<std::shared_ptr<T>> { };

  template<class T>
  struct impl<volatile std::shared_ptr<T>>: impl<std::shared_ptr<T>> { };

  template<class T>
  struct impl<const volatile std::shared_ptr<T>>: impl<std::shared_ptr<T>> { };
}}
using boost::get_pointer;
}



# else // if BOOST_VERSION < 105300

// Not standard conforming: add function to ::std(::tr1)
namespace std {

# if defined(_MSC_VER) && _MSC_VER < 1700
namespace tr1 {
# endif

    template<class T>
    T * get_pointer(shared_ptr<T> const& p) { return p.get(); }

# if defined(_MSC_VER) && _MSC_VER < 1700
} // namespace tr1
# endif

} // namespace std

#endif // if BOOST_VERSION < 105300 / else

namespace luabind {
    using std::get_deleter;

    template <typename T>
    struct default_converter<std::shared_ptr<T> >:
        detail::shared_ptr_converter<std::shared_ptr<T> > {};

    template <typename T>
    struct default_converter<std::shared_ptr<T> const&>:
        detail::shared_ptr_converter<std::shared_ptr<T> > {};
} // namespace luabind

#endif // if smart pointers are available

#endif // LUABIND_STD_SHAREDPTR_CONVERTER_HPP_INCLUDED
