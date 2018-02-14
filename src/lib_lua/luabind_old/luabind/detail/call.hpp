// Copyright Daniel Wallin 208.Use, modification and distribution is
// subject to the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef LUABIND_CALL2_080911_HPP
#define LUABIND_CALL2_080911_HPP

#include <luabind/config.hpp>
#include <typeinfo>
#include <luabind/detail/meta.hpp>
#include <luabind/detail/policy.hpp>
#include <luabind/yield_policy.hpp>
#include <luabind/detail/decorate_type.hpp>
#include <luabind/detail/object.hpp>

#ifdef LUABIND_NO_INTERNAL_TAG_ARGUMENTS
#include <tuple>
#endif

namespace luabind {
	namespace detail {

		struct invoke_context;

		struct LUABIND_API function_object
		{
			function_object(lua_CFunction entry)
				: entry(entry)
				, next(0)
			{}

			virtual ~function_object()
			{}

			virtual int call(lua_State* L, invoke_context& ctx) /* const */ = 0;
			virtual void format_signature(lua_State* L, char const* function) const = 0;

			lua_CFunction entry;
			std::string name;
			function_object* next;
			object keepalive;
		};

		struct LUABIND_API invoke_context
		{
			invoke_context()
				: best_score((std::numeric_limits<int>::max)())
				, candidate_index(0)
			{}

			operator bool() const
			{
				return candidate_index == 1;
			}

			void format_error(lua_State* L, function_object const* overloads) const;

			int best_score;
			function_object const* candidates[10];	// This looks like it could crash if you provide too many overloads?
			int candidate_index;
		};

		namespace call_detail_new {

			/*
				Compute Stack Indices
				Given the list of argument converter arities, computes the stack indices that each converter addresses.
			*/

			template< typename ConsumedList, unsigned int CurrentSum, unsigned int... StackIndices >
			struct compute_stack_indices;

			template< unsigned int Consumed0, unsigned int... Consumeds, unsigned int CurrentSum, unsigned int... StackIndices >
			struct compute_stack_indices< meta::index_list< Consumed0, Consumeds... >, CurrentSum, StackIndices... >
			{
				using type = typename compute_stack_indices< meta::index_list< Consumeds... >, CurrentSum + Consumed0, StackIndices..., CurrentSum >::type;
			};

			template< unsigned int CurrentSum, unsigned int... StackIndices >
			struct compute_stack_indices< meta::index_list< >, CurrentSum, StackIndices... >
			{
				using type = meta::index_list< StackIndices... >;
			};

			template< typename Foo >
			struct FooFoo {	// Foo!
				enum { consumed_args = Foo::consumed_args };
			};


			template< typename PolicyList, typename StackIndexList >
			struct policy_list_postcall;

			template< typename Policy0, typename... Policies, typename StackIndexList >
			struct policy_list_postcall< meta::type_list< call_policy_injector<Policy0>, Policies... >, StackIndexList > {
				static void postcall(lua_State* L, int results) {
					Policy0::postcall(L, results, StackIndexList());
					policy_list_postcall< meta::type_list< Policies... >, StackIndexList >::postcall(L, results);
				}
			};

			template< typename ConverterPolicy, typename StackIndexList, bool has_postcall >
			struct converter_policy_postcall {
				static void postcall(lua_State* L, int results) {
					ConverterPolicy::postcall(L, results, StackIndexList());
				}
			};

			template< typename ConverterPolicy, typename StackIndexList >
			struct converter_policy_postcall< ConverterPolicy, StackIndexList, false > {
				static void postcall(lua_State* /*L*/, int /*results*/) {
				}
			};

			template< unsigned int Index, typename Policy, typename... Policies, typename StackIndexList >
			struct policy_list_postcall< meta::type_list< converter_policy_injector< Index, Policy >, Policies... >, StackIndexList > {
				static void postcall(lua_State* L, int results) {
					converter_policy_postcall < Policy, StackIndexList, converter_policy_injector< Index, Policy >::has_postcall >::postcall(L, results);
					policy_list_postcall< meta::type_list< Policies... >, StackIndexList >::postcall(L, results);
				}
			};

			template< typename StackIndexList >
			struct policy_list_postcall< meta::type_list< >, StackIndexList > {
				static void postcall(lua_State* /*L*/, int /*results*/) {}
			};

#ifndef LUABIND_NO_INTERNAL_TAG_ARGUMENTS		
			template< typename... ArgumentConverters >
			struct compute_invoke_values {
				using consumed_list = meta::index_list< FooFoo<ArgumentConverters>::consumed_args... >;
				using stack_index_list = typename compute_stack_indices< consumed_list, 1 >::type;
				enum { arity = meta::sum<consumed_list>::value };
			};
#endif


		}

#ifndef LUABIND_NO_INTERNAL_TAG_ARGUMENTS
		inline int match_deferred(lua_State* L, meta::index_list<>, meta::type_list<>)
		{
			return 0;
		}

		template< unsigned int StackIndex0, unsigned int... StackIndices,
			typename ArgumentType0, typename... ArgumentTypes,
			typename ArgumentConverter0, typename... ArgumentConverters >
			int match_deferred(lua_State* L,
				meta::index_list< StackIndex0, StackIndices... >,
				meta::type_list< ArgumentType0, ArgumentTypes... >,
				ArgumentConverter0& converter0, ArgumentConverters&... converters
			)
		{
			const int this_match = converter0.match(L, decorated_type<ArgumentType0>(), StackIndex0);
			const int other_match = match_deferred(L, meta::index_list<StackIndices...>(), meta::type_list<ArgumentTypes...>(), converters...);
			return (this_match >= 0) ?	// could also sum them all up unconditionally
				this_match + match_deferred(L, meta::index_list<StackIndices...>(), meta::type_list<ArgumentTypes...>(), converters...)
				: no_match;
		}

		template< typename T, bool isvoid, bool memfun = std::is_member_function_pointer<T>::value > struct do_call_struct;

		template< typename T >
		struct do_call_struct< T, true, true /*memfun*/> {
			template< typename F, typename ArgumentType0, typename... ArgumentTypes, unsigned int StackIndex0, unsigned int... StackIndices, typename ReturnConverter, typename Argument0Converter, typename... ArgumentConverters >
			static void do_call(lua_State* L, F& f,
				meta::index_list<StackIndex0, StackIndices...>, meta::type_list<ArgumentType0, ArgumentTypes...>,
				ReturnConverter& result_converter, Argument0Converter& arg0_converter, ArgumentConverters&... arg_converters
			)
			{
				((arg0_converter.to_cpp(L, decorated_type<ArgumentType0>(), StackIndex0)).*f)(
					arg_converters.to_cpp(L, decorated_type<ArgumentTypes>(), StackIndices)...
					);
			}
		};

		template< typename T >
		struct do_call_struct< T, false, true /*memfun*/> {
			template< typename F, typename ArgumentType0, typename... ArgumentTypes, unsigned int StackIndex0, unsigned int... StackIndices, typename ReturnConverter, typename Argument0Converter, typename... ArgumentConverters >
			static void do_call(lua_State* L, F& f,
				meta::index_list<StackIndex0, StackIndices...>, meta::type_list<ArgumentType0, ArgumentTypes...>,
				ReturnConverter& result_converter, Argument0Converter& arg0_converter, ArgumentConverters&... arg_converters
			)
			{
				result_converter.to_lua(L,
					((arg0_converter.to_cpp(L, decorated_type<ArgumentType0>(), StackIndex0)).*f)(
						arg_converters.to_cpp(L, decorated_type<ArgumentTypes>(), StackIndices)...
						)
				);
			}
		};


		template< typename T >
		struct do_call_struct< T, true, false > {
			template<
				typename F,
				typename... ArgumentTypes, unsigned int... StackIndices,
				typename ReturnConverter, typename... ArgumentConverters
			>
				static void do_call(lua_State* L, F& f,
					meta::index_list<StackIndices...>, meta::type_list<ArgumentTypes...>,
					ReturnConverter& result_converter, ArgumentConverters&... arg_converters)
			{
				f(arg_converters.to_cpp(L, decorated_type<ArgumentTypes>(), StackIndices)...);
			}
		};

		template< typename T >
		struct do_call_struct< T, false, false > {
			template<
				typename F,
				typename... ArgumentTypes, unsigned int... StackIndices,
				typename ReturnConverter, typename... ArgumentConverters
			>
				static void do_call(lua_State* L, F& f,
					meta::index_list<StackIndices...>, meta::type_list<ArgumentTypes...>,
					ReturnConverter& result_converter, ArgumentConverters&... arg_converters)
			{
				result_converter.to_lua(L,
					f(arg_converters.to_cpp(L, decorated_type<ArgumentTypes>(), StackIndices)...)
				);
			}
		};

		template< typename F, typename ReturnType, typename... Arguments,
			typename ReturnConverter, typename... ArgumentConverters,
			unsigned int Index0, unsigned int... Indices, typename PolicyList
		>
			int invoke3(lua_State* L, function_object const& self, invoke_context& ctx, F& f,
				PolicyList, meta::index_list< Index0, Indices... > index_list, meta::type_list<ReturnType, Arguments...> signature_list,
				ReturnConverter return_converter, ArgumentConverters... argument_converters)
		{
			using invoke_values       = typename call_detail_new::compute_invoke_values< ArgumentConverters... >;
			using argument_list_type  = meta::type_list<Arguments...>;
			using argument_index_list = meta::index_list<Indices...>;

			int const arguments = lua_gettop(L);
			int score = no_match;

			if(invoke_values::arity == arguments) {
				score = match_deferred(L, typename invoke_values::stack_index_list(), argument_list_type(), argument_converters...);
			}

			if(score >= 0 && score < ctx.best_score) {
				ctx.best_score = score;
				ctx.candidates[0] = &self;
				ctx.candidate_index = 1;
			}
			else if(score == ctx.best_score) {
				ctx.candidates[ctx.candidate_index++] = &self;
			}

			int results = 0;

			if(self.next)
			{
				results = self.next->call(L, ctx);
			}

			if(score == ctx.best_score && ctx.candidate_index == 1)
			{
				do_call_struct<F, std::is_void<ReturnType>::value>::do_call(L, f, typename invoke_values::stack_index_list(), argument_list_type(), return_converter, argument_converters...);
				meta::init_order{ (argument_converters.converter_postcall(L, decorated_type<Arguments>(), meta::get< typename invoke_values::stack_index_list, Indices - 1 >::value), 0)... };

				results = lua_gettop(L) - invoke_values::arity;
				if(has_call_policy<PolicyList, yield_policy>::value) {
					results = lua_yield(L, results);
				}

				// call policiy list postcall
				call_detail_new::policy_list_postcall < PolicyList, typename meta::push_front< typename invoke_values::stack_index_list, meta::index<invoke_values::arity> >::type >::postcall(L, results);
			}

			return results;
		}

		template< typename F, typename ReturnType, typename... Arguments, unsigned int Index0, unsigned int... Indices, typename PolicyList >
		int invoke2(lua_State* L, function_object const& self, invoke_context& ctx, F& f,
			meta::type_list<ReturnType, Arguments...> signature, meta::index_list<Index0, Indices...>, PolicyList)
		{
			using signature_type   = meta::type_list<ReturnType, Arguments...>;
			using return_converter = specialized_converter_policy_n<0, PolicyList, ReturnType, cpp_to_lua>;
			return invoke3(L, self, ctx, f,
				PolicyList(), meta::index_list<Index0, Indices...>(), signature,
				return_converter(), specialized_converter_policy_n<Indices, PolicyList, Arguments, lua_to_cpp>()...
			);
		}


		template <class F, class Signature, typename... PolicyInjectors>
		// boost::bind's operator() is const, std::bind's is not
		inline int invoke(lua_State* L, function_object const& self, invoke_context& ctx, F& f, Signature,
			meta::type_list< PolicyInjectors... > const& injectors)
		{
			return invoke2(L, self, ctx, f, Signature(), typename meta::make_index_range<0, meta::size<Signature>::value>::type(), injectors);
		}
#endif

#ifdef LUABIND_NO_INTERNAL_TAG_ARGUMENTS

		// VC2013RC doesn't support expanding a template and its member template in one expression, that's why we have to to incrementally build
		// the converter list instead of a single combined expansion.
		template< typename ArgumentList, typename PolicyList, typename CurrentList = meta::type_list<>, unsigned int Counter = 1 >
		struct compute_argument_converter_list;

		template< typename Argument0, typename... Arguments, typename PolicyList, typename... CurrentConverters, unsigned int Counter >
		struct compute_argument_converter_list< meta::type_list<Argument0, Arguments... >, PolicyList, meta::type_list<CurrentConverters...>, Counter >
		{
			using converter_type   = typename policy_detail::get_converter_policy<Counter, PolicyList>::type;
			using this_specialized = typename converter_type::template specialize<Argument0, lua_to_cpp >::type;
			using type             = typename compute_argument_converter_list<meta::type_list<Arguments...>, PolicyList, meta::type_list<CurrentConverters..., this_specialized>, Counter + 1>::type;
		};

		template<typename PolicyList, typename... CurrentConverters, unsigned int Counter >
		struct compute_argument_converter_list< meta::type_list<>, PolicyList, meta::type_list<CurrentConverters...>, Counter >
		{
			using type = meta::type_list<CurrentConverters...>;
		};

		template< typename ConverterList >
		struct build_consumed_list;

		template< typename... Converters >
		struct build_consumed_list< meta::type_list< Converters... > > {
			using consumed_list = meta::index_list< call_detail_new::FooFoo<Converters>::consumed_args... >;
		};

		template< typename SignatureList, typename PolicyList >
		struct invoke_traits;

		// Specialization for free functions
		template< typename ResultType, typename... Arguments, typename PolicyList >
		struct invoke_traits< meta::type_list<ResultType, Arguments... >, PolicyList >
		{
			using signature_list   = meta::type_list<ResultType, Arguments...>;
			using policy_list      = PolicyList;
			using result_type      = ResultType;
			using result_converter = specialized_converter_policy_n<0, PolicyList, result_type, cpp_to_lua >;
			using argument_list    = meta::type_list<Arguments...>;

			using decorated_argument_list = meta::type_list< decorated_type<Arguments>... >;
			// note that this is 0-based, so whenever you want to fetch from the converter list, you need to add 1
			using argument_index_list           = typename meta::make_index_range< 0, sizeof...(Arguments) >::type;
			using argument_converter_list       = typename compute_argument_converter_list<argument_list, PolicyList>::type;
			using argument_converter_tuple_type = typename meta::make_tuple<argument_converter_list>::type;
			using consumed_list                 = typename build_consumed_list<argument_converter_list>::consumed_list;
			using stack_index_list              = typename call_detail_new::compute_stack_indices< consumed_list, 1 >::type;
			enum { arity = meta::sum<consumed_list>::value };
		};

		template< typename StackIndexList, typename SignatureList, unsigned int End = meta::size<SignatureList>::value, unsigned int Index = 1 >
		struct match_struct {
			template< typename TupleType >
			static int match(lua_State* L, TupleType& tuple)
			{
				const int this_match = std::get<Index - 1>(tuple).match(L, decorated_type<typename SignatureList::template at<Index>>(), meta::get<StackIndexList, Index - 1>::value);
				return this_match >= 0 ?	// could also sum them up unconditionally
					this_match + match_struct<StackIndexList, SignatureList, End, Index + 1>::match(L, tuple)
					: no_match;
			}
		};

		template< typename StackIndexList, typename SignatureList, unsigned int Index >
		struct match_struct< StackIndexList, SignatureList, Index, Index >
		{
			template< typename TupleType >
			static int match(lua_State* /*L*/, TupleType&) {
				return 0;
			}
		};

		template< typename PolicyList, typename Signature, typename F >
		struct invoke_struct
		{
			using traits = invoke_traits< Signature, PolicyList >;

			template< bool IsMember, bool IsVoid, typename IndexList >
			struct call_struct;

			template< unsigned int... ArgumentIndices >
			struct call_struct< false /*member*/, false /*void*/, meta::index_list<ArgumentIndices...> >
			{
				static void call(lua_State* L, F& f, typename traits::argument_converter_tuple_type& argument_tuple)
				{
					using decorated_list   = typename traits::decorated_argument_list;
					using stack_indices    = typename traits::stack_index_list;
					using result_converter = typename traits::result_converter;

					result_converter().to_lua(L,
						f((std::get<ArgumentIndices>(argument_tuple).to_cpp(L,
							typename meta::get<decorated_list, ArgumentIndices>::type(),
							meta::get<stack_indices, ArgumentIndices>::value))...
						)
					);

					meta::init_order{
						(std::get<ArgumentIndices>(argument_tuple).converter_postcall(L,
						typename meta::get<typename traits::decorated_argument_list, ArgumentIndices>::type(),
						meta::get<typename traits::stack_index_list, ArgumentIndices>::value), 0)...
					};
				}
			};

			template< unsigned int... ArgumentIndices >
			struct call_struct< false /*member*/, true /*void*/, meta::index_list<ArgumentIndices...> >
			{
				static void call(lua_State* L, F& f, typename traits::argument_converter_tuple_type& argument_tuple)
				{
					using decorated_list = typename traits::decorated_argument_list;
					using stack_indices  = typename traits::stack_index_list;

					// This prevents unused warnings with empty parameter lists
					(void)L;

					f(std::get<ArgumentIndices>(argument_tuple).to_cpp(L,
						typename meta::get<decorated_list, ArgumentIndices>::type(),
						meta::get<stack_indices, ArgumentIndices>::value)...

					);

					meta::init_order{
						(std::get<ArgumentIndices>(argument_tuple).converter_postcall(L,
						typename meta::get<typename traits::decorated_argument_list, ArgumentIndices>::type(),
						meta::get<typename traits::stack_index_list, ArgumentIndices>::value), 0)...
					};
				}
			};

			template< unsigned int ClassIndex, unsigned int... ArgumentIndices >
			struct call_struct< true /*member*/, false /*void*/, meta::index_list<ClassIndex, ArgumentIndices...> >
			{
				static void call(lua_State* L, F& f, typename traits::argument_converter_tuple_type& argument_tuple)
				{
					using decorated_list   = typename traits::decorated_argument_list;
					using stack_indices    = typename traits::stack_index_list;
					using result_converter = typename traits::result_converter;

					auto& object = std::get<0>(argument_tuple).to_cpp(L,
						typename meta::get<typename traits::decorated_argument_list, 0>::type(), 1);

					result_converter().to_lua(L,
						(object.*f)(std::get<ArgumentIndices>(argument_tuple).to_cpp(L,
							typename meta::get<decorated_list, ArgumentIndices>::type(),
							meta::get<stack_indices, ArgumentIndices>::value)...
							)
					);

					meta::init_order{
						(std::get<ArgumentIndices>(argument_tuple).converter_postcall(L,
						typename meta::get<typename traits::decorated_argument_list, ArgumentIndices>::type(),
						meta::get<typename traits::stack_index_list, ArgumentIndices>::value), 0)...
					};
				}
			};

			template< unsigned int ClassIndex, unsigned int... ArgumentIndices >
			struct call_struct< true /*member*/, true /*void*/, meta::index_list<ClassIndex, ArgumentIndices...> >
			{
				static void call(lua_State* L, F& f, typename traits::argument_converter_tuple_type& argument_tuple)
				{
					using decorated_list = typename traits::decorated_argument_list;
					using stack_indices  = typename traits::stack_index_list;

					auto& object = std::get<0>(argument_tuple).to_cpp(L, typename meta::get<typename traits::decorated_argument_list, 0>::type(), 1);

					(object.*f)(std::get<ArgumentIndices>(argument_tuple).to_cpp(L,
						typename meta::get<decorated_list, ArgumentIndices>::type(),
						meta::get<stack_indices, ArgumentIndices>::value)...
						);

					meta::init_order{
						(std::get<ArgumentIndices>(argument_tuple).converter_postcall(L,
						typename meta::get<typename traits::decorated_argument_list, ArgumentIndices>::type(),
						meta::get<typename traits::stack_index_list, ArgumentIndices>::value), 0)...
					};
				}
			};

			static int invoke(lua_State* L, function_object const& self, invoke_context& ctx, F& f) {
				int const arguments = lua_gettop(L);
				int score = no_match;

				// Even match needs the tuple, since pointer_converters buffer the cast result
				typename traits::argument_converter_tuple_type converter_tuple;

				if(traits::arity == arguments) {
					// Things to remember:
					// 0 is the perfect match. match > 0 means that objects had to be casted, where the value
					// is the total distance of all arguments to their given types (graph distance).
					// This is why we can say MaxArguments = 100, MaxDerivationDepth = 100, so no match will be > 100*100=10k and -10k1 absorbs every match.
					// This gets rid of the awkward checks during converter match traversal.
					using struct_type = match_struct< typename traits::stack_index_list, typename traits::signature_list >;
					score = struct_type::match(L, converter_tuple);
				}

				if(score >= 0 && score < ctx.best_score) {
					ctx.best_score = score;
					ctx.candidates[0] = &self;
					ctx.candidate_index = 1;
				}
				else if(score == ctx.best_score) {
					ctx.candidates[ctx.candidate_index++] = &self;
				}

				int results = 0;

				if(self.next)
				{
					results = self.next->call(L, ctx);
				}

				if(score == ctx.best_score && ctx.candidate_index == 1)
				{
					call_struct<
						std::is_member_function_pointer<F>::value,
						std::is_void<typename traits::result_type>::value,
						typename traits::argument_index_list
					>::call(L, f, converter_tuple);

					results = lua_gettop(L) - traits::arity;
					if(has_call_policy<PolicyList, yield_policy>::value) {
						results = lua_yield(L, results);
					}

					call_detail_new::policy_list_postcall < PolicyList, typename meta::push_front< typename traits::stack_index_list, meta::index<traits::arity> >::type >::postcall(L, results);
				}

				return results;
			}

		};

		template< typename PolicyList, typename Signature, typename F>
		inline int invoke(lua_State* L, function_object const& self, invoke_context& ctx, F& f)
		{
			return invoke_struct<PolicyList, Signature, F>::invoke(L, self, ctx, f);
		}
#endif

	}
} // namespace luabind::detail

# endif // LUABIND_CALL2_080911_HPP

