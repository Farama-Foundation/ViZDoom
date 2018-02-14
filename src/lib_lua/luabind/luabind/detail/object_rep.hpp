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


#ifndef LUABIND_OBJECT_REP_HPP_INCLUDED
#define LUABIND_OBJECT_REP_HPP_INCLUDED

#include <luabind/config.hpp>
#include <luabind/detail/class_rep.hpp>
#include <luabind/detail/instance_holder.hpp>

#include <boost/aligned_storage.hpp>

namespace luabind { namespace detail
{
    // implements the selection between dynamic dispatch
    // or default implementation calls from within a virtual
    // function wrapper. The input is the self reference on
    // the top of the stack. Output is the function to call
    // on the top of the stack (the input self reference will
    // be popped)
    LUABIND_API void do_call_member_selection(lua_State* L, char const* name);

    void finalize(lua_State* L, class_rep* crep);

    // this class is allocated inside lua for each pointer.
    // it contains the actual c++ object-pointer.
    // it also tells if it is const or not.
    class LUABIND_API object_rep
    {
    public:
        object_rep(instance_holder* instance, class_rep* crep);
        ~object_rep();

        const class_rep* crep() const { return m_classrep; }
        class_rep* crep() { return m_classrep; }

        void set_instance(instance_holder* instance) { m_instance = instance; }

        void add_dependency(lua_State* L, int index);
        void release_dependency_refs(lua_State* L);

        std::pair<void*, int> get_instance(class_id target) const
        {
            if (m_instance == 0)
                return std::pair<void*, int>(static_cast<void*>(0), -1);
            return m_instance->get(m_classrep->casts(), target);
        }

        bool is_const() const
        {
            return m_instance && m_instance->pointee_const();
        }

        void release()
        {
            if (m_instance)
                m_instance->release();
        }

        void* allocate(std::size_t size)
        {
            if (size <= instance_buffer_size)
                return &m_instance_buffer;
            return std::malloc(size);
        }

        void deallocate(void* storage)
        {
            if (storage == &m_instance_buffer)
                return;
            std::free(storage);
        }

    private:

        object_rep(object_rep const&);

        void operator=(object_rep const&);

        BOOST_STATIC_CONSTANT(std::size_t, instance_buffer_size=32);
        boost::aligned_storage<instance_buffer_size> m_instance_buffer;
        instance_holder* m_instance;
        class_rep* m_classrep; // the class information about this object's type
        std::size_t m_dependency_cnt; // counts dependencies
    };

    template<class T>
    struct delete_s
    {
        static void apply(void* ptr)
        {
            delete static_cast<T*>(ptr);
        }
    };

    template<class T>
    struct destruct_only_s
    {
        static void apply(void* ptr)
        {
            // Removes unreferenced formal parameter warning on VC7.
            (void)ptr;
#ifndef NDEBUG
            int completeness_check[sizeof(T)];
            (void)completeness_check;
#endif
            static_cast<T*>(ptr)->~T();
        }
    };

    LUABIND_API object_rep* get_instance(lua_State* L, int index);
    LUABIND_API void push_instance_metatable(lua_State* L);
    LUABIND_API object_rep* push_new_instance(lua_State* L, class_rep* cls);

}}

#endif // LUABIND_OBJECT_REP_HPP_INCLUDED
