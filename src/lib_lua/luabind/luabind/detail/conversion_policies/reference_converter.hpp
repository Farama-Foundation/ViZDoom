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

#ifndef LUABIND_REFERENCE_CONVERTER_HPP_INCLUDED
#define LUABIND_REFERENCE_CONVERTER_HPP_INCLUDED

#include <type_traits>
#include <luabind/detail/conversion_policies/conversion_base.hpp>
#include <luabind/detail/conversion_policies/pointer_converter.hpp>
#include <luabind/back_reference.hpp>

namespace luabind {

	/*
		TODO: Remove code duplication
	*/
	namespace detail {

		struct ref_converter : pointer_converter
		{
			using type      = ref_converter;
			using is_native = std::false_type;

			enum { consumed_args = 1 };

			template<class T>
			void to_lua(lua_State* L, T& ref)
			{
				if(luabind::get_back_reference(L, ref))
					return;

				make_pointee_instance(L, ref, std::false_type());
			}

			template<class T>
			T& to_cpp(lua_State* L, by_reference<T>, int index)
			{
				assert(!lua_isnil(L, index));
				return *pointer_converter::to_cpp(L, by_pointer<T>(), index);
			}

			template<class T>
			int match(lua_State* L, by_reference<T>, int index)
			{
				object_rep* obj = get_instance(L, index);
				if(obj == 0) return no_match;

				if(obj->is_const())
					return no_match;

				std::pair<void*, int> s = obj->get_instance(registered_class<T>::id);
				result = s.first;
				return s.second;
			}

			template<class T>
			void converter_postcall(lua_State*, T, int) {}
		};

		struct const_ref_converter
		{
			using type      = const_ref_converter;
			using is_native = std::false_type;

			enum { consumed_args = 1 };

			const_ref_converter()
				: result(0)
			{}

			void* result;

			template<class T>
			void to_lua(lua_State* L, T const& ref)
			{
				if(luabind::get_back_reference(L, ref))
					return;

				make_pointee_instance(L, ref, std::false_type());
			}

			template<class T>
			T const& to_cpp(lua_State*, by_const_reference<T>, int)
			{
				return *static_cast<T*>(result);
			}

			template<class T>
			int match(lua_State* L, by_const_reference<T>, int index)
			{
				object_rep* obj = get_instance(L, index);
				if(obj == 0) return no_match; // if the type is not one of our own registered types, classify it as a non-match

				std::pair<void*, int> s = obj->get_instance(registered_class<T>::id);
				if(s.second >= 0 && !obj->is_const())
					s.second += 10;
				result = s.first;
				return s.second;
			}

			template<class T>
			void converter_postcall(lua_State*, by_const_reference<T>, int)
			{
			}
		};

	}
}

#endif

