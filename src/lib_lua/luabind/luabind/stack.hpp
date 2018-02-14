#ifndef LUABIND_STACK_HPP_INCLUDED
#define LUABIND_STACK_HPP_INCLUDED

#include <luabind/detail/convert_to_lua.hpp>
#include <luabind/detail/policy.hpp>
#include <luabind/detail/primitives.hpp>
#include <luabind/error.hpp>

#include <boost/implicit_cast.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/apply_wrap.hpp>
#include <boost/optional/optional.hpp>

namespace luabind
{
namespace detail
{
  namespace mpl = boost::mpl;

  template<class T, class ConverterGenerator>
  void push_aux(lua_State* L, T& value, ConverterGenerator*)
  {
      typedef typename boost::mpl::if_<
          boost::is_reference_wrapper<T>
        , BOOST_DEDUCED_TYPENAME boost::unwrap_reference<T>::type&
        , T
      >::type unwrapped_type;

      typename mpl::apply_wrap2<
          ConverterGenerator,unwrapped_type,cpp_to_lua
      >::type cv;

      cv.apply(
          L
        , boost::implicit_cast<
              BOOST_DEDUCED_TYPENAME boost::unwrap_reference<T>::type&
          >(value)
      );
  }

} // namespace detail

template<class T, class Policies>
void push(lua_State* L, T& value, Policies const&)
{
    typedef typename detail::find_conversion_policy<
        0
      , Policies
    >::type converter_policy;

    push_aux(L, value, static_cast<converter_policy*>(0));
}

template<class T>
void push(lua_State* L, T& value)
{
    push(L, value, detail::null_type());
}

namespace detail
{

  template<
      class T
    , class Policies
    , class ErrorPolicy
    , class ReturnType
  >
  ReturnType from_lua_aux(
      lua_State* L
    , int idx
    , T*
    , Policies*
    , ErrorPolicy*
    , ReturnType*
  )
  {
#ifndef LUABIND_NO_ERROR_CHECKING
      if (!L)
          return ErrorPolicy::handle_error(L, typeid(void));
#endif
      typedef typename detail::find_conversion_policy<
          0
        , Policies
      >::type converter_generator;

      typename mpl::apply_wrap2<converter_generator, T, lua_to_cpp>::type cv;

      if (cv.match(L, LUABIND_DECORATE_TYPE(T), idx) < 0)
      {
          return ErrorPolicy::handle_error(L, typeid(T));
      }

      return cv.apply(L, LUABIND_DECORATE_TYPE(T), idx);
  }

# ifdef BOOST_MSVC
#  pragma warning(push)
#  pragma warning(disable:4702) // unreachable code
# endif

  template<class T>
  struct throw_error_policy
  {
      static T handle_error(lua_State* L, type_id const& type_info)
      {
#ifndef LUABIND_NO_EXCEPTIONS
          throw cast_failed(L, type_info);
#else
          cast_failed_callback_fun e = get_cast_failed_callback();
          if (e) e(L, type_info);

          assert(0 && "object_cast failed. If you want to handle this error use "
              "luabind::set_error_callback()");
          std::terminate();
          return *(typename boost::remove_reference<T>::type*)0;
#endif
      }
  };

# ifdef BOOST_MSVC
#  pragma warning(pop)
# endif

  template<class T>
  struct nothrow_error_policy
  {
      static boost::optional<T> handle_error(lua_State*, type_id const&)
      {
          return boost::optional<T>();
      }
  };

} // namespace detail


template<class T, class Policies>
T from_lua(lua_State* L, int idx, Policies const&)
{
    return detail::from_lua_aux(
        L
      , idx
      , static_cast<T*>(0)
      , static_cast<Policies*>(0)
      , static_cast<detail::throw_error_policy<T>*>(0)
      , static_cast<T*>(0)
    );
}

template<class T>
T from_lua(lua_State* L, int idx)
{
    return from_lua<T>(L, idx, detail::null_type());
}

template<class T, class Policies>
boost::optional<T> from_lua_nothrow(lua_State* L, int idx, Policies const&)
{
    return detail::from_lua_aux(
        L
      , idx
      , static_cast<T*>(0)
      , static_cast<Policies*>(0)
      , static_cast<detail::nothrow_error_policy<T>*>(0)
      , static_cast<boost::optional<T>*>(0)
    );
}

template<class T>
boost::optional<T> from_lua_nothrow(lua_State* L, int idx)
{
    return from_lua_nothrow<T>(L, idx, detail::null_type());
}


} // namespace luabind

#endif // LUABIND_STACK_HPP_INCLUDED
