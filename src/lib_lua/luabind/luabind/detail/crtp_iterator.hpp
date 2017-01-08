#ifndef LUABIND_CRTP_ITERATOR_HPP_INCLUDED
#define LUABIND_CRTP_ITERATOR_HPP_INCLUDED
#include <iterator>

namespace luabind {
	namespace detail {

		template< typename CRTP, typename Category, typename ValueType, typename ReferenceType = ValueType&, typename DifferenceType = ptrdiff_t >
		class crtp_iterator :
			public std::iterator<Category, ValueType, DifferenceType, ValueType*, ReferenceType >
		{
		public:
			using base_type = std::iterator<Category, ValueType, DifferenceType, ValueType*, ReferenceType >;


			CRTP& operator++()
			{
				upcast().increment();
				return upcast();
			}

			CRTP operator++(int)
			{
				CRTP tmp(upcast());
				upcast().increment();
				return tmp;
			}

			bool operator==(const CRTP& rhs)
			{
				return upcast().equal(rhs);
			}

			bool operator!=(const CRTP& rhs)
			{
				return !upcast().equal(rhs);
			}

			typename base_type::reference operator*()
			{
				return upcast().dereference();
			}

			typename base_type::reference operator->()
			{
				return upcast().dereference();
			}

		private:
			CRTP& upcast() { return static_cast<CRTP&>(*this); }
			const CRTP& upcast() const { return static_cast<const CRTP&>(*this); }
		};

	}
}






#endif

