// Copyright Christian Neumüller 2013. Use, modification and distribution is
// subject to the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !BOOST_PP_IS_ITERATING

#   ifndef LUABIND_FUNCTION_CONVERTER_HPP_INCLUDED
#   define LUABIND_FUNCTION_CONVERTER_HPP_INCLUDED

#   include <luabind/detail/call_function.hpp>
#   include <luabind/back_reference.hpp>
#   include <luabind/make_function.hpp>
#   include <luabind/object.hpp>

#   include <boost/preprocessor/repetition/enum_trailing.hpp>

namespace luabind {

    template <typename R = object>
    struct function
    {
        typedef R result_type;

        function(luabind::object const& obj)
            : m_func(obj)
        {
        }

        R operator() ()
        {
            return call_function<R>(m_func);
        }

#   define BOOST_PP_ITERATION_LIMITS (1, LUABIND_MAX_ARITY)
#   define BOOST_PP_FILENAME_1 <luabind/function_converter.hpp> // include self
#   include BOOST_PP_ITERATE()

    private:
        object m_func;
    };

    template <>
    struct function<void>
    {
        typedef void result_type;

        function(luabind::object const& obj)
            : m_func(obj)
        {
        }

        void operator() ()
        {
            call_function<void>(m_func);
        }

#   define VOID_SPEC
#   define BOOST_PP_ITERATION_LIMITS (1, LUABIND_MAX_ARITY)
#   define BOOST_PP_FILENAME_1 <luabind/function_converter.hpp> // include self
#   include BOOST_PP_ITERATE()
#   undef VOID_SPEC

    private:
        luabind::object m_func;
    };

    template <typename F>
    struct default_converter<F,
            typename boost::enable_if<detail::is_function<F> >::type>
    {
        typedef boost::mpl::true_ is_native;

        int consumed_args() const
        {
            return 1;
        }

        template <class U>
        void converter_postcall(lua_State*, U const&, int)
        {}

        template <class U>
        static int match(lua_State* L, U, int index)
        {
            if (lua_type(L, index) == LUA_TFUNCTION)
                return 0;
            if (luaL_getmetafield(L, index, "__call")) {
                lua_pop(L, 1);
                return 1;
            }
            return -1;
        }

        template <class U>
        F apply(lua_State* L, U, int index)
        {
            // If you get a compiler error here, you are probably trying to
            // get a function pointer from Lua. This is not supported:
            // you must use a type which is constructible from a
            // luabind::function, e.g. std::function or boost::function.
            return function<typename F::result_type>(
                object(from_stack(L, index)));
        }

        void apply(lua_State* L, F value)
        {
            make_function(L, value).push(L);
        }
    };
} // namespace luabind

#   endif // LUABIND_FUNCTION_CONVERTER_HPP_INCLUDED

#else  // !BOOST_PP_IS_ITERATING

#   define N BOOST_PP_ITERATION()

#   define TMPL_PARAMS   BOOST_PP_ENUM_PARAMS(N, typename A)
#   ifndef LUABIND_NO_RVALUE_REFERENCES
#       define TYPED_ARGS BOOST_PP_ENUM_BINARY_PARAMS(N, A, && a)
#   else
#       define TYPED_ARGS BOOST_PP_ENUM_BINARY_PARAMS(N, A, const& a)
#   endif
#   ifndef LUABIND_NO_RVALUE_REFERENCES
#       define PRINT_FORWARD_ARG(z, n, _) std::forward<BOOST_PP_CAT(A, n)>(BOOST_PP_CAT(a, n))
#       define TRAILING_ARGS BOOST_PP_ENUM_TRAILING(N, PRINT_FORWARD_ARG, ~)
#   else
#       define TRAILING_ARGS BOOST_PP_ENUM_TRAILING_PARAMS(N, a)
#   endif

#   ifdef VOID_SPEC

    template <TMPL_PARAMS>
    void operator() (TYPED_ARGS)
    {
        luabind::call_function<void>(m_func TRAILING_ARGS);
    }

#   else

    template <TMPL_PARAMS>
    R operator() (TYPED_ARGS)
    {
        return luabind::call_function<R>(m_func TRAILING_ARGS);
    }

#   endif

#   undef TMPL_PARAMS
#   undef TYPED_ARGS
#   undef TRAILING_ARGS
#   undef N

#endif // BOOST_PP_IS_ITERATING
