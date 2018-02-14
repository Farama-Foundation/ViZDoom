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


#ifndef LUABIND_CLASS_HPP_INCLUDED
#define LUABIND_CLASS_HPP_INCLUDED

/*
	ISSUES:
	------------------------------------------------------

	* solved for member functions, not application operator *
	if we have a base class that defines a function a derived class must be able to
	override that function (not just overload). Right now we just add the other overload
	to the overloads list and will probably get an ambiguity. If we want to support this
	each method_rep must include a vector of type_info pointers for each parameter.
	Operators do not have this problem, since operators always have to have
	it's own type as one of the arguments, no ambiguity can occur. Application
	operator, on the other hand, would have this problem.
	Properties cannot be overloaded, so they should always be overridden.
	If this is to work for application operator, we really need to specify if an application
	operator is const or not.

	If one class registers two functions with the same name and the same
	signature, there's currently no error. The last registered function will
	be the one that's used.
	How do we know which class registered the function? If the function was
	defined by the base class, it is a legal operation, to override it.
	we cannot look at the pointer offset, since it always will be zero for one of the bases.



	TODO:
	------------------------------------------------------

	finish smart pointer support
		* the adopt policy should not be able to adopt pointers to held_types. This
		must be prohibited.
		* name_of_type must recognize holder_types and not return "custom"

	document custom policies, custom converters

	store the instance object for policies.

	support the __concat metamethod. This is a bit tricky, since it cannot be
	treated as a normal operator. It is a binary operator but we want to use the
	__tostring implementation for both arguments.

*/

#include <luabind/prefix.hpp>
#include <luabind/config.hpp>

#include <string>
#include <map>
#include <vector>
#include <cassert>

#include <luabind/config.hpp>
#include <luabind/scope.hpp>
#include <luabind/back_reference.hpp>
#include <luabind/function.hpp>	// -> object.hpp
#include <luabind/dependency_policy.hpp>
#include <luabind/detail/constructor.hpp>	// -> object.hpp
#include <luabind/detail/deduce_signature.hpp>
#include <luabind/detail/primitives.hpp>
#include <luabind/detail/property.hpp>
#include <luabind/detail/typetraits.hpp>
#include <luabind/detail/class_rep.hpp>
#include <luabind/detail/object_rep.hpp>
#include <luabind/detail/call.hpp>
#include <luabind/detail/call_member.hpp>
#include <luabind/detail/enum_maker.hpp>
#include <luabind/detail/operator_id.hpp>
#include <luabind/detail/pointee_typeid.hpp>
#include <luabind/detail/link_compatibility.hpp>
#include <luabind/detail/inheritance.hpp>
#include <luabind/detail/signature_match.hpp>
#include <luabind/no_dependency.hpp>
#include <luabind/typeid.hpp>
#include <luabind/detail/meta.hpp>

// to remove the 'this' used in initialization list-warning
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4355)
#endif

namespace luabind {
	namespace detail {
		struct unspecified {};

		template<class Derived> struct operator_;

		struct you_need_to_define_a_get_const_holder_function_for_your_smart_ptr {};
	}

	template < typename... BaseClasses >
	struct bases { };

	using no_bases = bases< >;
	using default_holder = detail::null_type;

	namespace detail {

		template < typename T >
		struct make_bases {
			using type = bases< T >;
		};

		template< typename... Bases >
		struct make_bases< bases< Bases... > > {
			using type = bases< Bases... >;
		};
	}

	template< typename T >
	using make_bases = typename detail::make_bases< T >::type;

	template< typename... Args >
	struct constructor
	{};

	// helper for overloaded methods, only need to provide argument types (IntelliSense bug squiggles the code, but it does compile!)
	template< typename... Args >
	struct meth {
		template< typename Class, typename Ret >
		static auto use_nonconst(Ret(Class::*fn)(Args...)) -> decltype(fn)
		{
			return fn;
		}

		template< typename Class, typename Ret >
		static auto use_const(Ret(Class::*fn)(Args...) const) -> decltype(fn)
		{
			return fn;
		}

		template< typename Class, typename Ret >
		static auto use_auto(Ret(Class::*fn)(Args...) const) -> decltype(fn)
		{
			return fn;
		}

		template< typename Class, typename Ret >
		static auto use_auto(Ret(Class::*fn)(Args...)) -> decltype(fn)
		{
			return fn;
		}
	};

	// TODO: Could specialize for certain base classes to make the interface "type safe".
	template<typename T, typename BaseOrBases = no_bases, typename HolderType = detail::null_type, typename WrapperType = detail::null_type>
	struct class_;

	// TODO: this function will only be invoked if the user hasn't defined a correct overload
	// maybe we should have a static assert in here?
	inline detail::you_need_to_define_a_get_const_holder_function_for_your_smart_ptr*
		get_const_holder(...)
	{
		return 0;
	}

	template <class T>
	std::shared_ptr<T const>* get_const_holder(std::shared_ptr<T>*)
	{
		return 0;
	}



	namespace detail {
		// prints the types of the values on the stack, in the
		// range [start_index, lua_gettop()]

		LUABIND_API std::string stack_content_by_name(lua_State* L, int start_index);

		struct LUABIND_API create_class
		{
			static int stage1(lua_State* L);
			static int stage2(lua_State* L);
		};

	} // detail

	namespace detail {

		template<class T>
		struct static_scope
		{
			static_scope(T& self_) : self(self_)
			{
			}

			T& operator[](scope s) const
			{
				self.add_inner_scope(s);
				return self;
			}

		private:
			template<class U> void operator,(U const&) const;
			void operator=(static_scope const&);

			T& self;
		};

		struct class_registration;

		struct LUABIND_API class_base : scope
		{
		public:
			class_base(char const* name);

			struct base_desc
			{
				type_id type;
				int ptr_offset;
			};

			void init(
				type_id const& type, class_id id
				, type_id const& wrapped_type, class_id wrapper_id);

			void add_base(type_id const& base, cast_function cast);

			void add_member(registration* member);
			void add_default_member(registration* member);

			const char* name() const;

			void add_static_constant(const char* name, int val);
			void add_inner_scope(scope& s);

			void add_cast(class_id src, class_id target, cast_function cast);

		private:
			class_registration* m_registration;
		};

		// MSVC complains about member being sensitive to alignment (C4121)
		// when F is a pointer to member of a class with virtual bases.
# ifdef _MSC_VER
#  pragma pack(push)
#  pragma pack(16)
# endif

		template <class Class, class F, class Policies>
		struct memfun_registration : registration
		{
			memfun_registration(char const* name, F f)
				: name(name), f(f)
			{}

			void register_(lua_State* L) const
			{
				// Need to check if the class type of the signature is a base of this class
				object fn = make_function(L, f, typename call_types< F, Class >::signature_type(), Policies());
				add_overload(object(from_stack(L, -1)), name, fn);
			}

			char const* name;
			F f;
		};

# ifdef _MSC_VER
#  pragma pack(pop)
# endif

		template <class P, class T>
		struct default_pointer
		{
			using type = P;
		};

		template <class T>
		struct default_pointer<null_type, T>
		{
			using type = std::unique_ptr<T>;
		};

		template <class Class, class Pointer, class Signature, class Policies>
		struct constructor_registration : registration
		{
			constructor_registration()
			{}

			void register_(lua_State* L) const
			{
				using pointer = typename default_pointer<Pointer, Class>::type;
				object fn = make_function(L, construct<Class, pointer, Signature>(), Signature(), Policies());
				add_overload(object(from_stack(L, -1)), "__init", fn);
			}
		};

		template <class T>
		struct reference_result
			: std::conditional< std::is_pointer<T>::value || is_primitive<T>::value, T, typename std::add_lvalue_reference< T >::type >
		{};

		template <class T>
		struct reference_argument
			: std::conditional< std::is_pointer<T>::value || is_primitive<T>::value, T, typename std::add_lvalue_reference< typename std::add_const<T>::type >::type >
		{};

		template <class T, class Policies>
		struct inject_dependency_policy
		{
			using type = typename std::conditional <
				is_primitive<T>::value || meta::contains<Policies, call_policy_injector< detail::no_dependency_policy > >::value,
				Policies,
				typename meta::push_back< Policies, call_policy_injector< dependency_policy<0, 1> > >::type
			>::type;
		};

		template <class Class, class Get, class GetPolicies, class Set = null_type, class SetPolicies = no_policies >
		struct property_registration : registration
		{
			property_registration(char const* name, Get const& get, Set const& set = detail::null_type())
				: name(name), get(get), set(set)
			{}

			template <class F>
			object make_get(lua_State* L, F const& f, std::false_type /*member_ptr*/) const
			{
				return make_function(L, f, GetPolicies());
			}

			template <class T, class D>
			object make_get(lua_State* L, D T::* mem_ptr, std::true_type /*member_ptr*/) const
			{
				using result_type = typename reference_result<D>::type;
				using get_signature = meta::type_list<result_type, Class const&>;
				using injected_list = typename inject_dependency_policy< D, GetPolicies >::type;

				return make_function(L, access_member_ptr<T, D, result_type>(mem_ptr), get_signature(), injected_list());
			}

			template <class F>
			object make_set(lua_State* L, F const& f, std::false_type /*member_ptr*/) const
			{
				return make_function(L, f, typename call_types< F >::signature_type(), SetPolicies());
			}

			template <class T, class D>
			object make_set(lua_State* L, D T::* mem_ptr, std::true_type /*member_ptr*/) const
			{
				using argument_type  = typename reference_argument<D>::type;
				using signature_type = meta::type_list<void, Class&, argument_type>;

				return make_function(L, access_member_ptr<T, D>(mem_ptr), signature_type(), SetPolicies());
			}

			// if a setter was given
			template <class SetterType>
			void register_aux(lua_State* L, object const& context, object const& get_, SetterType const&) const
			{
				context[name] = property(get_, make_set(L, set, std::is_member_object_pointer<Set>()));
			}

			// if no setter was given
			void register_aux(lua_State*, object const& context, object const& get_, null_type) const
			{
				context[name] = property(get_);
			}

			// register entry
			void register_(lua_State* L) const
			{
				object context(from_stack(L, -1));
				register_aux(L, context, make_get(L, get, std::is_member_object_pointer<Get>()), set);
			}


			char const* name;
			Get get;
			Set set;
		};

	} // namespace detail

	// registers a class in the lua environment
	template<class T, typename BaseOrBases, typename HolderType, typename WrapperType >
	struct class_
		: detail::class_base
	{
		using self_t = class_<T, BaseOrBases, HolderType, WrapperType>;
		using BaseList = make_bases< BaseOrBases >;


	public:
		class_(const char* name) : class_base(name), scope(*this)
		{
#ifndef NDEBUG
			detail::check_link_compatibility();
#endif
			init();
		}

		// virtual functions
		template<class F, typename... Injectors>
		class_& def(char const* name, F fn, policy_list< Injectors... > policies = no_policies())
		{
			return this->virtual_def(name, fn, policies, detail::null_type());
		}

		// IntelliSense bug squiggles the code, but it does compile!
		template<typename Ret, typename C, typename... Args, typename... Injectors>
		class_& def_nonconst(char const* name, Ret(C::*fn)(Args...), policy_list<Injectors...> policies = no_policies())
		{
			return def(name, fn, policies);
		}

		// IntelliSense bug squiggles the code, but it does compile!
		template<typename Ret, typename C, typename... Args, typename... Injectors>
		class_& def_const(char const* name, Ret(C::*fn)(Args...) const, policy_list<Injectors...> policies = no_policies())
		{
			return def(name, fn, policies);
		}

		template<class F, class Default, typename... Injectors>
		class_& def(char const* name, F fn, Default default_, policy_list< Injectors... > policies = no_policies())
		{
			return this->virtual_def(name, fn, policies, default_);
		}

		template<typename... Args, typename... Injectors>
		class_& def(constructor<Args...> sig, policy_list< Injectors... > policies = no_policies())
		{
			return this->def_constructor(sig, policies);
		}

		// ======================
		// Start of reworked property overloads
		// ======================

		template <class Getter, typename... Injectors>
		class_& property(const char* name, Getter g, policy_list< Injectors... > get_injectors = no_policies())
		{
			return property(name, g, detail::null_type(), get_injectors);
		}

		template <class Getter, class Setter, typename... GetInjectors, typename... SetInjectors>
		class_& property(const char* name, Getter g, Setter s, policy_list<GetInjectors...> = no_policies(), policy_list<SetInjectors...> = no_policies())
		{
			using registration_type = detail::property_registration<T, Getter, policy_list<GetInjectors...>, Setter, policy_list<SetInjectors...>>;
			this->add_member(new registration_type(name, g, s));
			return *this;
		}

		template <class C, class D, typename... Injectors>
		class_& def_readonly(const char* name, D C::*mem_ptr, policy_list<Injectors...> policies = no_policies())
		{
			return property(name, mem_ptr, policies);
		}

		template <class C, class D, typename... GetInjectors, typename... SetInjectors>
		class_& def_readwrite(const char* name, D C::*mem_ptr, policy_list<GetInjectors...> get_injectors = no_policies(), policy_list<SetInjectors...> set_injectors = no_policies())
		{
			return property(name, mem_ptr, mem_ptr, get_injectors, set_injectors);
		}

		// =====================
		// End of reworked property overloads
		// =====================

		template<class Derived, typename... Injectors>
		class_& def(detail::operator_<Derived>, policy_list<Injectors...> policies = no_policies())
		{
			using policy_list_type = policy_list<Injectors...>;
			return this->def(Derived::name(), &Derived::template apply<T, policy_list_type>::execute, policies);
		}

		detail::enum_maker<self_t> enum_(const char*)
		{
			return detail::enum_maker<self_t>(*this);
		}

		detail::static_scope<self_t> scope;

	private:
		void init()
		{
			class_base::init(typeid(T), detail::registered_class<T>::id, typeid(WrapperType), detail::registered_class<WrapperType>::id);
			add_wrapper_cast((WrapperType*)0);
			generate_baseclass_list();
		}


		template<class S, typename OtherBaseOrBases, typename OtherWrapper >
		class_(const class_<S, OtherBaseOrBases, OtherWrapper>&);

		template <class Src, class Target>
		void add_downcast(Src*, Target*, std::true_type)
		{
			add_cast(detail::registered_class<Src>::id, detail::registered_class<Target>::id, detail::dynamic_cast_<Src, Target>::execute);
		}

		template <class Src, class Target>
		void add_downcast(Src*, Target*, std::false_type)
		{}

		// this function generates conversion information
		// in the given class_rep structure. It will be able
		// to implicitly cast to the given template type
		template<typename Class0, typename... Classes>
		void gen_base_info(bases<Class0, Classes...>)
		{
			add_base(typeid(Class0), detail::static_cast_<T, Class0>::execute);
			add_cast(detail::registered_class<T>::id, detail::registered_class<Class0>::id, detail::static_cast_<T, Class0>::execute);
			add_downcast((Class0*)0, (T*)0, std::is_polymorphic<Class0>());
			gen_base_info(bases<Classes...>());
		}

		void gen_base_info(bases<>)
		{
		}

		void generate_baseclass_list()
		{
			gen_base_info(BaseList());
		}

		void operator=(class_ const&);

		void add_wrapper_cast(detail::null_type*)
		{}

		template <class U>
		void add_wrapper_cast(U*)
		{
			add_cast(detail::registered_class<U>::id, detail::registered_class<T>::id, detail::static_cast_<U, T>::execute);
			add_downcast((T*)0, (U*)0, std::is_polymorphic<T>());
		}

		// these handle default implementation of virtual functions
		template<class F, class Default, typename... Injectors>
		class_& virtual_def(char const* name, F const& fn, policy_list< Injectors... >, Default default_)
		{
			using policy_list_type = policy_list< Injectors... >;
			this->add_member(new detail::memfun_registration<T, F, policy_list_type      >(name, fn));
			this->add_default_member(new detail::memfun_registration<T, Default, policy_list_type>(name, default_));
			return *this;
		}

		template<class F, typename... Injectors>
		class_& virtual_def(char const* name, F const& fn, policy_list< Injectors... >, detail::null_type)
		{
			using policy_list_type = policy_list< Injectors... >;
			this->add_member(new detail::memfun_registration<T, F, policy_list_type>(name, fn));
			return *this;
		}

		template<typename... SignatureElements, typename... Injectors>
		class_& def_constructor(constructor<SignatureElements...> const&, policy_list< Injectors... > const&)
		{
			using signature_type = meta::type_list<void, argument const&, SignatureElements...>;
			using policy_list_type = policy_list< Injectors... >;

			using construct_type = typename std::conditional<
				detail::is_null_type<WrapperType>::value,
				T,
				WrapperType
			>::type;

			using registration_type = detail::constructor_registration<construct_type, HolderType, signature_type, policy_list_type>;
			this->add_member(new registration_type());
			this->add_default_member(new registration_type());

			return *this;
		}
	};

}

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#endif // LUABIND_CLASS_HPP_INCLUDED

