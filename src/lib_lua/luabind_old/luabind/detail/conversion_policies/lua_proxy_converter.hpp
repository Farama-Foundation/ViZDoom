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

#ifndef LUABIND_VALUE_WRAPPER_CONVERTER_HPP_INCLUDED
#define LUABIND_VALUE_WRAPPER_CONVERTER_HPP_INCLUDED

#include <luabind/lua_proxy.hpp>
#include <type_traits>

namespace luabind {
	namespace detail {

		template <class U>
		struct lua_proxy_converter
		{
			using type      = lua_proxy_converter<U>;
			using is_native = std::true_type;

			enum { consumed_args = 1 };

			template<class T>
			T to_cpp(lua_State* L, by_const_reference<T>, int index)
			{
				return T(from_stack(L, index));
			}

			template<class T>
			T to_cpp(lua_State* L, by_value<T>, int index)
			{
				return to_cpp(L, by_const_reference<T>(), index);
			}

			template<class T>
			static int match(lua_State* L, by_const_reference<T>, int index)
			{
				return lua_proxy_traits<T>::check(L, index)
					? max_hierarchy_depth
					: no_match;
			}

			template<class T>
			static int match(lua_State* L, by_value<T>, int index)
			{
				return match(L, by_const_reference<T>(), index);
			}

			void converter_postcall(...) {}

			template<class T>
			void to_lua(lua_State* interpreter, T const& value_wrapper)
			{
				lua_proxy_traits<T>::unwrap(interpreter, value_wrapper);
			}
		};

	}
}

#endif

