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

#ifndef LUABIND_CALL_SHARED_HPP_INCLUDED
#define LUABIND_CALL_SHARED_HPP_INCLUDED

namespace luabind {
	namespace detail {

		inline void call_error(lua_State* L)
		{
#ifndef LUABIND_NO_EXCEPTIONS
			throw luabind::error(L);
#else
			error_callback_fun e = get_error_callback();
			if(e) e(L);

			assert(0 && "the lua function threw an error and exceptions are disabled."
				" If you want to handle the error you can use luabind::set_error_callback()");
			std::terminate();
#endif
		}

		template<typename T>
		void cast_error(lua_State* L)
		{
#ifndef LUABIND_NO_EXCEPTIONS
			throw cast_failed(L, typeid(T));
#else
			cast_failed_callback_fun e = get_cast_failed_callback();
			if(e) e(L, typeid(T));

			assert(0 && "the lua function's return value could not be converted."
				" If you want to handle the error you can use luabind::set_cast_failed_callback()");
			std::terminate();
#endif	
		}

		template< typename... Args >
		void expand_hack(Args... args)
		{}

	}
}

#endif