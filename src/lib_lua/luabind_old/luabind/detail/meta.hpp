// Copyright Michael Steinberg 2013. Use, modification and distribution is
// subject to the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef LUABIND_META_HPP_INCLUDED
#define LUABIND_META_HPP_INCLUDED

#include <tuple>

namespace luabind {
	namespace meta {

		struct type_list_tag {};
		struct index_list_tag {};

		/*
		Index list and type list share pretty common patterns... is there a way to unify them?
		*/

		template< unsigned int >
		struct count
		{};

		template< unsigned int >
		struct index
		{};

		template< typename Type >
		struct type
		{};

		// Use this to unpack a parameter pack into a list of T's
		template< typename T, typename DontCare >
		struct unpack_helper
		{
			using type = T;
		};

		struct init_order {
			init_order(std::initializer_list<int>) {}
		};

		// common operators
		template< typename T >
		struct size;

		template< typename T, unsigned int Index >	// Specializations so index lists can use the same syntax
		struct get;

		template< typename... Lists >
		struct join;

		template< typename List1, typename List2, typename... Lists >
		struct join< List1, List2, Lists... > {
			// Could try to join on both sides
			using type = typename join< typename join< List1, List2 >::type, Lists... >::type;
		};

		// convenience
		template< typename T >
		struct front : public get< T, 0 >
		{
		};

		template< typename List, typename T >
		struct push_front;

		template< typename List, typename T >
		struct push_back;

		template< typename T >
		struct pop_front;

		template< typename T >
		struct pop_back;

		template< typename List, unsigned int Index, typename T >
		struct replace;

		template< typename List, unsigned int Index, template< typename > class T >
		struct enwrap;

		template< typename List, template< typename > class T >
		struct enwrap_all;

		template< typename List, unsigned int Index, template< typename > class T >
		struct transform;

		template< typename List, unsigned int Index, template< typename > class Function >
		using transform_t = typename transform< List, Index, Function >::type;

		template< typename List, template< typename > class Function >
		struct transform_all;

		template< typename List, template< typename > class Function >
		using transform_all_t = typename transform_all< List, Function >::type;


		template< typename T, unsigned int start, unsigned int end >
		struct sub_range;

		/*
		aliases
		*/
		template< typename T >
		using pop_front_t = typename pop_front<T>::type;

		template< typename T1, typename... Types >
		using join_t = typename join< T1, Types... >::type;

		template< typename T, unsigned int start, unsigned int end >
		using sub_range_t = typename sub_range< T, start, end >::type;

		template< typename T, unsigned int index >
		using get_t = typename get< T, index >::type;


		// Used as terminator on type and index lists
		struct null_type {};

		template< typename... Types >
		struct type_list : public type_list_tag
		{
			template< unsigned int Index >
			using at = typename get<type_list, Index>::type;
		};

		template< typename... Types1, typename... Types2 >
		type_list<Types1..., Types2...> operator|(const type_list<Types1...>&, const type_list<Types2...>&) {
			return type_list<Types1..., Types2...>();
		}

		template< typename T >
		struct is_typelist : public std::false_type
		{
			static const bool value = false;
		};

		template< typename... Types >
		struct is_typelist< type_list< Types... > > : public std::true_type
		{
			static const bool value = true;
		};

		/*
		Find type
		*/

		template< typename TypeList, typename Type >
		struct contains;

		template< typename Type0, typename... Types, typename Type >
		struct contains< type_list<Type0, Types...>, Type >
			: std::conditional< std::is_same<Type0, Type>::value, std::true_type, contains< type_list<Types...>, Type > >::type
		{
		};

		template< typename Type >
		struct contains< type_list< >, Type >
			: std::false_type
		{
		};

		/*
		size
		*/

		template< >
		struct size< type_list< > > {
			enum { value = 0 };
		};

		template< typename Type0, typename... Types >
		struct size< type_list< Type0, Types... > > {
			enum { value = 1 + size< type_list<Types...> >::value };
		};


		template< typename... Types, typename Type >
		struct push_front< type_list<Types...>, Type >
		{
			using type = type_list< Type, Types... >;
		};

		template< typename... Types, typename Type >
		struct push_back< type_list<Types...>, Type >
		{
			using type = type_list< Types..., Type >;
		};

		/*
		pop_front
		*/

		template< typename Type0, typename... Types >
		struct pop_front< type_list< Type0, Types... > >
		{
			using type = type_list< Types... >;
		};

		template< >
		struct pop_front< type_list< > >
		{
			using type = type_list< >;
		};

		/*
		Index access to type list
		*/
		template< typename Element0, typename... Elements, unsigned int Index >
		struct get< type_list<Element0, Elements...>, Index > {
			using type = typename get< type_list<Elements...>, Index - 1 >::type;
		};

		template< typename Element0, typename... Elements >
		struct get< type_list<Element0, Elements...>, 0 >
		{
			using type = Element0;
		};

		template< unsigned int Index >
		struct get< type_list< >, Index >
		{
			static_assert(size< type_list< int > >::value == 1, "Bad Index");
		};

		/*
		Join Type Lists
		*/
		template< typename... Types1, typename... Types2 >
		struct join< type_list< Types1... >, type_list< Types2... > >
		{
			using type = type_list< Types1..., Types2... >;
		};

		namespace detail {
			template< typename HeadList, typename TailList, typename Type, unsigned int Index >
			struct replace_helper;

			template< typename... HeadTypes, typename CurrentType, typename... TailTypes, typename Type, unsigned int Index >
			struct replace_helper< type_list< HeadTypes... >, type_list< CurrentType, TailTypes... >, Type, Index> {
				using type = typename replace_helper< type_list< HeadTypes..., CurrentType >, type_list<TailTypes...>, Type, Index - 1 >::type;
			};

			template< typename... HeadTypes, typename CurrentType, typename... TailTypes, typename Type >
			struct replace_helper< type_list< HeadTypes... >, type_list< CurrentType, TailTypes... >, Type, 0 > {
				using type = type_list< HeadTypes..., Type, TailTypes... >;
			};
		}

		template< typename... Types, unsigned int Index, typename Type >
		struct replace< type_list< Types... >, Index, Type >
		{
			using TypeList = type_list< Types... >;

			using type = join_t<
				sub_range_t< TypeList, 0, Index >,
				meta::type_list<Type>,
				sub_range_t< TypeList, Index + 1, sizeof...(Types) >
			>;
		};

		/*
		Enwrap all elements of a type list in an template
		*/
		template< typename... Types, unsigned int Index, template< typename >  class Enwrapper >
		struct enwrap< type_list< Types... >, Index, Enwrapper > {
			using type = join_t<
				sub_range_t< type_list<Types...>, 0, Index >,
				Enwrapper< get_t< type_list<Types...>, Index> >,
				sub_range_t< type_list<Types...>, Index + 1, sizeof...(Types) >
			>;
		};

		template< typename... Types, template< typename > class Enwrapper >
		struct enwrap_all< type_list< Types... >, Enwrapper >
		{
			using type = type_list< Enwrapper< Types >... >;
		};

		/*
		Transform a certain element of a type list
		*/
		template< typename T, unsigned int Index, template< typename > class Function >
		struct transform;

		template< typename... Types, unsigned int Index, template< typename > class Function >
		struct transform< type_list< Types... >, Index, Function > {
			using type = join_t<
				sub_range_t< type_list<Types...>, 0, Index >,
				typename Function< get_t< type_list<Types...>, Index> >::type,
				sub_range_t< type_list<Types...>, Index + 1, sizeof...(Types) >
			>;
		};

		/*
		Transform all elements of a type list
		*/
		template< typename... Types, template< typename >  class Function >
		struct transform_all< type_list< Types... >, Function > {
			using type = type_list< typename Function<Types>::type... >;
		};

		/*
		Tuple from type list
		*/
		template< class TypeList >
		struct make_tuple;

		template< typename... Types >
		struct make_tuple< type_list< Types... > >
		{
			using type = std::tuple< Types... >;
		};


		/*
		Type selection
		*/

		template< typename ConvertibleToTrueFalse, typename Result >
		struct case_ : public ConvertibleToTrueFalse {
			using type = Result;
		};

		template< typename Result >
		struct default_ {
			using type = Result;
		};


		template< typename Case, typename... CaseList >
		struct select_
		{
			using type = typename std::conditional<
				std::is_convertible<Case, std::true_type>::value,
				typename Case::type,
				typename select_<CaseList...>::type
			>::type;
		};

		template< typename Case >
		struct select_< Case >
		{
			using type = typename std::conditional<
				std::is_convertible<Case, std::true_type>::value,
				typename Case::type,
				null_type
			>::type;
		};

		template< typename T >
		struct select_< default_<T> > {
			using type = typename default_<T>::type;
		};

		/*
		Create index lists to expand on type_lists
		*/

		template< unsigned int... Indices >
		struct index_list : public index_list_tag
		{
		};

		/*
		Index index list
		*/

		namespace detail {

			template< unsigned int Index, unsigned int Value0, unsigned int... Values >
			struct get_iterate {
				static const unsigned int value = get_iterate< Index - 1, Values... >::value;
			};

			template< unsigned int Value0, unsigned int... Values >
			struct get_iterate< 0, Value0, Values... > {
				static const unsigned int value = Value0;
			};

		}

		template< unsigned int... Values, unsigned int Index >
		struct get< index_list< Values... >, Index >
		{
			static_assert(sizeof...(Values) > Index, "Bad Index");
			static const unsigned int value = detail::get_iterate< Index, Values... >::value;
		};

		/*
		Index list size
		*/

		template< unsigned int... Values >
		struct size< index_list< Values... > > {
			static const unsigned int value = sizeof...(Values);
		};


		/*
		Index list push front
		*/
		template< unsigned int... Indices, unsigned int Index >
		struct push_front< index_list< Indices... >, index<Index> >
		{
			using type = index_list< Index, Indices... >;
		};

		/*
		Index list push back
		*/
		template< unsigned int... Indices, unsigned int Index >
		struct push_back< index_list< Indices... >, index<Index> >
		{
			using type = index_list< Indices..., Index >;
		};

		/*
		Index list pop_front
		*/
		template< unsigned int Index0, unsigned int... Indices >
		struct pop_front< index_list< Index0, Indices... > > {
			using type = index_list< Indices... >;
		};

		template< >
		struct pop_front< index_list< > > {
			using type = index_list<  >;
		};

		/*
		Index list range creation
		*/
		namespace detail {

			template< unsigned int curr, unsigned int end, unsigned int... Indices >
			struct make_index_range :
				public make_index_range< curr + 1, end, Indices..., curr >
			{
			};

			template< unsigned int end, unsigned int... Indices >
			struct make_index_range< end, end, Indices... >
			{
				using type = index_list< Indices... >;
			};

		}

		/*
			make_index_range< start, end >
			Creates the index list list of range [start, end)
		*/
		template< unsigned int start, unsigned int end >
		struct make_index_range {
			static_assert(end >= start, "end must be greater than or equal to start");
			using type = typename detail::make_index_range< start, end >::type;
		};

		template< unsigned int start, unsigned int end >
		using index_range = typename make_index_range<start, end>::type;

		namespace detail {
			// These implementation are not really efficient...
			template< typename SourceList, typename IndexList >
			struct sub_range_index;

			template< typename SourceList, unsigned int... Indices >
			struct sub_range_index< SourceList, index_list< Indices... > > {
				using type = index_list< get< SourceList, Indices >::value... >;
			};

			template< typename SourceList, typename IndexList >
			struct sub_range_type;

			template< typename SourceList, unsigned int... Indices >
			struct sub_range_type< SourceList, index_list< Indices... > > {
				using type = type_list< typename get< SourceList, Indices >::type... >;
			};

		}

		/*
		Index list sub_range [start, end)
		*/
		template< unsigned int start, unsigned int end, unsigned int... Indices >
		struct sub_range< index_list<Indices...>, start, end >
		{
			static_assert(end >= start, "end must be greater or equal to start");
			using type = typename detail::sub_range_index< index_list<Indices...>, typename make_index_range<start, end>::type >::type;
		};

		/*
		Type list sub_range [start, end)
		*/
		template< unsigned int start, unsigned int end, typename... Types >
		struct sub_range< type_list<Types...>, start, end >
		{
			static_assert(end >= start, "end must be greater or equal to start");
			using type = typename detail::sub_range_type< type_list<Types...>, typename make_index_range<start, end>::type >::type;
		};

		/*
		Index list sum
		*/

		namespace detail {

			template< typename T, T... Values >
			struct sum_values;

			template< typename T, T Value0, T... Values >
			struct sum_values< T, Value0, Values... > {
				static const T value = Value0 + sum_values< T, Values... >::value;
			};

			template< typename T >
			struct sum_values< T >
			{
				static const T value = 0;
			};

		}

		template< typename T >
		struct sum;

		template< unsigned int... Args >
		struct sum< index_list<Args...> >
		{
			static const unsigned int value = detail::sum_values<unsigned int, Args...>::value;
		};

		/*
			and_ or_
		*/

		template< typename... ConvertiblesToTrueFalse >
		struct and_;

		template< typename Convertible0, typename... Convertibles >
		struct and_< Convertible0, Convertibles... >
			: std::conditional <
			std::is_convertible< Convertible0, std::true_type >::value,
			and_< Convertibles... >,
			std::false_type > ::type
		{
		};

		template< >
		struct and_< >
			: std::true_type
		{
		};


		template< typename... ConvertiblesToTrueFalse >
		struct or_;

		template< typename Convertible0, typename... Convertibles >
		struct or_< Convertible0, Convertibles... >
			: std::conditional <
			std::is_convertible< Convertible0, std::true_type >::value,
			std::true_type,
			or_< Convertibles... >
			> ::type
		{
		};

		template< >
		struct or_< >
			: std::true_type
		{
		};

	}
}

#endif

