// Copyright Daniel Wallin 2008. Use, modification and distribution is
// subject to the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef LUABIND_TYPEID_081227_HPP
# define LUABIND_TYPEID_081227_HPP

# include <luabind/detail/primitives.hpp>

# include <boost/operators.hpp>

# include <cstdlib>
# include <cstring>
# include <string>
# include <typeinfo>

// boost/units/detail/utility.hpp
# if defined(__GLIBCXX__) || defined(__GLIBCPP__)
#  define LUABIND_USE_DEMANGLING
#  include <cxxabi.h>
# endif // __GNUC__

// See https://svn.boost.org/trac/boost/ticket/754
# if (defined(__GNUC__) && __GNUC__ >= 3) \
      || defined(_AIX) \
      || (defined(__sgi) && defined(__host_mips)) \
      || (defined(__hpux) && defined(__HP_aCC)) \
      || (defined(linux) && defined(__INTEL_COMPILER) && defined(__ICC))
#  define LUABIND_SAFE_TYPEID
# endif

namespace luabind {

# ifdef BOOST_MSVC
#  pragma warning(push)
// std::type_info::before() returns int, rather than bool.
// At least on MSVC7.1, this is true for the comparison
// operators as well.
#  pragma warning(disable:4800)
# endif

class type_id
  : boost::less_than_comparable<type_id>
{
public:
    type_id()
      : m_id(&typeid(detail::null_type))
    {}

    type_id(std::type_info const& id)
      : m_id(&id)
    {}

    bool operator!=(type_id const& other) const
    {
# ifdef LUABIND_SAFE_TYPEID
        return std::strcmp(m_id->name(), other.m_id->name()) != 0;
# else
        return *m_id != *other.m_id;
# endif
    }

    bool operator==(type_id const& other) const
    {
# ifdef LUABIND_SAFE_TYPEID
        return std::strcmp(m_id->name(), other.m_id->name()) == 0;
# else
        return *m_id == *other.m_id;
# endif
    }

    bool operator<(type_id const& other) const
    {
# ifdef LUABIND_SAFE_TYPEID
        return std::strcmp(m_id->name(), other.m_id->name()) < 0;
# else
        return m_id->before(*other.m_id);
# endif
    }

    std::string name() const
    {
# ifdef LUABIND_USE_DEMANGLING
        int status;
        char* buf = abi::__cxa_demangle(m_id->name(), 0, 0, &status);
        if (buf != 0) {
            std::string demangled_name(buf);
            std::free(buf);
            return demangled_name;
        }
# endif
        return m_id->name();
    }

private:
    std::type_info const* m_id;
};

# ifdef BOOST_MSVC
#  pragma warning(pop)
# endif

} // namespace luabind

#endif // LUABIND_TYPEID_081227_HPP
