// Copyright Daniel Wallin 2009. Use, modification and distribution is
// subject to the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef LUABIND_SHARED_PTR_CONVERTER_090211_HPP
# define LUABIND_SHARED_PTR_CONVERTER_090211_HPP

# include <luabind/detail/decorate_type.hpp>  // for LUABIND_DECORATE_TYPE
# include <luabind/detail/policy.hpp>    // for default_converter, etc
# include <luabind/detail/yes_no.hpp>
# include <luabind/get_main_thread.hpp>  // for get_main_thread
# include <luabind/handle.hpp>           // for handle

# include <boost/mpl/bool.hpp>           // for bool_, false_
# include <boost/smart_ptr/shared_ptr.hpp>  // for shared_ptr, get_deleter

namespace luabind {

namespace mpl = boost::mpl;

namespace detail
{

  LUABIND_API extern char state_use_count_tag;

  struct LUABIND_API shared_ptr_deleter
  {
      shared_ptr_deleter(lua_State* L, int index)
        : life_support(get_main_thread(L), L, index)
      {
          alter_use_count(L, +1);
      }

      void operator()(void const*)
      {
          lua_State* L = life_support.interpreter();
          assert(L);
          handle().swap(life_support);
          alter_use_count(L, -1);
      }

      handle life_support;

  private:
      static void alter_use_count(lua_State* L, lua_Integer diff);
  };

  // From http://stackoverflow.com/a/1007175/2128694
  // (by Johannes Schaub - litb, "based on a brilliant idea of someone on
  // usenet")
  template <typename T>
  struct has_shared_from_this_aux
  {
      has_shared_from_this_aux();  // not implemented; silence Clang warning
  private:

      struct fallback { int shared_from_this; }; // introduce member name
      struct derived : T, fallback
      {
          derived(); // not implemented; silence MSVC warnings C4510 and C4610
      };

      template<typename C, C> struct check;

      template<typename C> static no_t f(
          check<int fallback::*, &C::shared_from_this>*);
      template<typename C> static yes_t f(...);

  public:
      BOOST_STATIC_CONSTANT(bool, value =
          sizeof(f<derived>(0)) == sizeof(yes_t));
    };

  template <typename T>
  struct has_shared_from_this:
      mpl::bool_<has_shared_from_this_aux<T>::value>
  {};

} // namespace detail

using boost::get_deleter;

namespace detail {

template <class P>
struct shared_ptr_converter
  : default_converter<typename P::element_type*>
{
private:
    typedef P ptr_t;
    typedef typename ptr_t::element_type* rawptr_t;
    detail::value_converter m_val_cv;
    int m_val_score;

    // no shared_from_this() available
    ptr_t shared_from_raw(rawptr_t raw, lua_State* L, int index, mpl::false_)
    {
        return ptr_t(raw, detail::shared_ptr_deleter(L, index));
    }

    // shared_from_this() available.
    ptr_t shared_from_raw(rawptr_t raw, lua_State*, int, mpl::true_)
    {
        return ptr_t(raw->shared_from_this(), raw);
    }

public:
    shared_ptr_converter(): m_val_score(-1) {}

    typedef mpl::false_ is_native;

    template <class U>
    int match(lua_State* L, U, int index)
    {
        // Check if the value on the stack is a holder with exactly ptr_t
        // as pointer type.
        m_val_score = m_val_cv.match(L, LUABIND_DECORATE_TYPE(ptr_t), index);
        if (m_val_score >= 0)
            return m_val_score;

        // Fall back to raw_ptr.
        return default_converter<rawptr_t>::match(
            L, LUABIND_DECORATE_TYPE(rawptr_t), index);
    }

    template <class U>
    ptr_t apply(lua_State* L, U, int index)
    {
        // First, check if we got away without upcasting.
        if (m_val_score >= 0)
        {
            ptr_t ptr = m_val_cv.apply(
                L, LUABIND_DECORATE_TYPE(ptr_t), index);
            return ptr;
        }

        // If not obtain, a raw pointer and construct are shared one from it.
        rawptr_t raw_ptr = default_converter<rawptr_t>::apply(
            L, LUABIND_DECORATE_TYPE(rawptr_t), index);
        if (!raw_ptr)
            return ptr_t();

        return shared_from_raw(
            raw_ptr, L, index,
            detail::has_shared_from_this<typename ptr_t::element_type>());
    }

    void apply(lua_State* L, ptr_t const& p)
    {
        if (detail::shared_ptr_deleter* d =
                get_deleter<detail::shared_ptr_deleter>(p)) // Rely on ADL.
        {
            d->life_support.push(L);
        }
        else
        {
            m_val_cv.apply(L, p);
        }
    }

    template <class U>
    void converter_postcall(lua_State*, U const&, int)
    {}
};
} // namepace detail

template <typename T>
struct default_converter<boost::shared_ptr<T> >:
    detail::shared_ptr_converter<boost::shared_ptr<T> > {};

template <typename T>
struct default_converter<boost::shared_ptr<T> const&>:
    detail::shared_ptr_converter<boost::shared_ptr<T> > {};

typedef void(*state_unreferenced_fun)(lua_State*);

LUABIND_API void set_state_unreferenced_callback(
    lua_State* L, state_unreferenced_fun cb);
LUABIND_API state_unreferenced_fun get_state_unreferenced_callback(
    lua_State* L);
LUABIND_API bool is_state_unreferenced(lua_State* L);

} // namespace luabind

#endif // LUABIND_SHARED_PTR_CONVERTER_090211_HPP
