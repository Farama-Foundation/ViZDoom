// Copyright Daniel Wallin 2008. Use, modification and distribution is
// subject to the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef LUABIND_TAG_FUNCTION_081129_HPP
# define LUABIND_TAG_FUNCTION_081129_HPP

# include <luabind/detail/deduce_signature.hpp>

# if LUABIND_MAX_ARITY <= 8
#  include <boost/mpl/vector/vector10.hpp>
# else
#  include <boost/mpl/vector/vector50.hpp>
# endif
# include <boost/preprocessor/cat.hpp>
# include <boost/preprocessor/iterate.hpp>
# include <boost/preprocessor/repetition/enum_params.hpp>
# include <boost/preprocessor/repetition/enum_trailing_params.hpp>

# include <luabind/lua_state_fwd.hpp>

namespace luabind {

namespace detail
{

  struct invoke_context;
  struct function_object;

  template <class Signature, class F>
  struct tagged_function
  {
      tagged_function(F f_)
        : f(f_)
      {}

      F f;
  };

  template <class Signature, class F>
  struct signature_aux<tagged_function<Signature, F> >
  {
      typedef Signature type;
  };

  template <class Signature, class F, class SignatureSeq, class Policies>
  int invoke(
      lua_State* L, function_object const& self, invoke_context& ctx
    , tagged_function<Signature, F> const& tagged
    , SignatureSeq, Policies const& policies)
  {
      return invoke(L, self, ctx, tagged.f, SignatureSeq(), policies);
  }

} // namespace detail

template <class Signature, class F>
detail::tagged_function<Signature, F> tag_function(F f)
{
    return f;
}

} // namespace luabind

#endif // LUABIND_TAG_FUNCTION_081129_HPP
