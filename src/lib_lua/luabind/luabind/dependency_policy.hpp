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


#ifndef LUABIND_DEPENDENCY_POLICY_HPP_INCLUDED
#define LUABIND_DEPENDENCY_POLICY_HPP_INCLUDED

#include <luabind/config.hpp>
#include <luabind/detail/object_rep.hpp>  // for object_rep
#include <luabind/detail/policy.hpp>    // for policy_cons, etc
#include <luabind/detail/primitives.hpp>  // for null_type

namespace luabind { namespace detail
{
    // makes A dependent on B, meaning B will outlive A.
    // internally A stores a reference to B
    template<int A, int B>
    struct dependency_policy
    {
        static void postcall(lua_State* L, const index_map& indices)
        {
            int nurse_index = indices[A];
            int patient = indices[B];

            object_rep* nurse = static_cast<object_rep*>(lua_touserdata(L, nurse_index));

            // If the nurse isn't an object_rep, just make this a nop.
            if (nurse == 0)
                return;

            nurse->add_dependency(L, patient);
        }
    };

}}

namespace luabind
{
    template<int A, int B>
    detail::policy_cons<detail::dependency_policy<A, B>, detail::null_type>
    dependency(LUABIND_PLACEHOLDER_ARG(A), LUABIND_PLACEHOLDER_ARG(B))
    {
        return detail::policy_cons<detail::dependency_policy<A, B>, detail::null_type>();
    }

    template<int A>
    detail::policy_cons<detail::dependency_policy<0, A>, detail::null_type>
    return_internal_reference(LUABIND_PLACEHOLDER_ARG(A))
    {
        return detail::policy_cons<detail::dependency_policy<0, A>, detail::null_type>();
    }
}

#endif // LUABIND_DEPENDENCY_POLICY_HPP_INCLUDED
