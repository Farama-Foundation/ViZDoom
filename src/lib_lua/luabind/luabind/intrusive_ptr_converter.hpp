// Copyright Chsritian Neumüller 2013. Use, modification and distribution is
// subject to the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef LUABIND_INTRUSIVE_PTR_CONVERTER_HPP_INCLUDED
#define LUABIND_INTRUSIVE_PTR_CONVERTER_HPP_INCLUDED

#include <luabind/config.hpp>
#include <luabind/detail/decorate_type.hpp>
#include <luabind/detail/policy.hpp>

#include <boost/mpl/bool.hpp>
#include <boost/smart_ptr/intrusive_ptr.hpp>

namespace luabind {
namespace detail {
    template <class P>
struct intrusive_ptr_converter
  : default_converter<typename P::element_type*>
{
private:
    typedef P ptr_t;
    typedef typename ptr_t::element_type* rawptr_t;

public:
    typedef mpl::false_ is_native;

    template <class U>
    int match(lua_State* L, U, int index)
    {
        return default_converter<rawptr_t>::match(
            L, LUABIND_DECORATE_TYPE(rawptr_t), index);
    }

    template <class U>
    ptr_t apply(lua_State* L, U, int index)
    {
        rawptr_t raw_ptr = default_converter<rawptr_t>::apply(
            L, LUABIND_DECORATE_TYPE(rawptr_t), index);
        return ptr_t(raw_ptr);
    }

    void apply(lua_State* L, ptr_t const& p)
    {
        detail::value_converter().apply(L, p);
    }

    template <class U>
    void converter_postcall(lua_State*, U const&, int)
    {}
};
} // namespace detail

template <typename T>
struct default_converter<boost::intrusive_ptr<T> >:
    detail::intrusive_ptr_converter<boost::intrusive_ptr<T> > {};

template <typename T>
struct default_converter<boost::intrusive_ptr<T> const&>:
    detail::intrusive_ptr_converter<boost::intrusive_ptr<T> > {};

} // namespace luabind

#endif // LUABIND_INTRUSIVE_PTR_CONVERTER_HPP_INCLUDED
