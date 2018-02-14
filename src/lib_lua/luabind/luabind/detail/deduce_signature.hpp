// Copyright Daniel Wallin 2008. Use, modification and distribution is
// subject to the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# ifndef LUABIND_DEDUCE_SIGNATURE_080911_HPP
#  define LUABIND_DEDUCE_SIGNATURE_080911_HPP

#  include <luabind/detail/most_derived.hpp>

#  include <boost/function_types/components.hpp>
#  include <boost/function_types/is_callable_builtin.hpp>
#  include <boost/function_types/is_member_function_pointer.hpp>
#  include <boost/type_traits/add_reference.hpp>
#  include <boost/utility/enable_if.hpp>

namespace luabind { namespace detail {

namespace mpl = boost::mpl;
namespace fty = boost::function_types;

template <class F, class Enable=void>
struct signature_aux
{
    typedef F type;
};

// Handle boost::function and std::function:
template <template<class> class F, class S>
struct signature_aux<
    F<S>,
    typename mpl::apply< // Enable only if F<S>::result_type exists.
        mpl::always<void>, typename F<S>::result_type>::type >
{
    typedef S type;
};

template <class F>
struct signature_aux<const F>: signature_aux<F> {};

template <typename F>
struct is_function:
    fty::is_callable_builtin<typename signature_aux<F>::type>
{};

template <class F>
typename boost::enable_if<
    is_function<F>,
    fty::components<typename signature_aux<F>::type> >::type
deduce_signature(F const&, ...)
{
    return fty::components<typename signature_aux<F>::type>();
}

template <class F, class Wrapped>
typename boost::enable_if<
    fty::is_member_function_pointer<typename signature_aux<F>::type>,
    fty::components<
        typename signature_aux<F>::type,
        boost::add_reference<most_derived<mpl::_1, Wrapped> > > >::type
deduce_signature(F, Wrapped*)
{
    return fty::components<
        typename signature_aux<F>::type,
        boost::add_reference<most_derived<mpl::_1, Wrapped> > >();
}

} } // namespace luabind::detail

#endif  // LUABIND_DEDUCE_SIGNATURE_080911_HPP
