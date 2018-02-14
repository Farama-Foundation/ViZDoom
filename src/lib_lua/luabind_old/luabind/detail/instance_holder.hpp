// Copyright Daniel Wallin 2008. Use, modification and distribution is
// subject to the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef LUABIND_INSTANCE_HOLDER_081024_HPP
# define LUABIND_INSTANCE_HOLDER_081024_HPP

# include <luabind/detail/inheritance.hpp>
# include <luabind/pointer_traits.hpp>
# include <luabind/typeid.hpp>
# include <stdexcept>

namespace luabind {
	namespace detail {

		class instance_holder
		{
		public:
			instance_holder(bool pointee_const)
				: m_pointee_const(pointee_const)
			{}

			virtual ~instance_holder()
			{}

			virtual std::pair<void*, int> get(cast_graph const& casts, class_id target) const = 0;
			virtual void release() = 0;

			bool pointee_const() const
			{
				return m_pointee_const;
			}

		private:
			bool m_pointee_const;
		};

		template <class P, class Pointee = void const>
		class pointer_holder : public instance_holder
		{
		public:
			pointer_holder(P p, class_id dynamic_id, void* dynamic_ptr) :
				instance_holder(detail::is_pointer_to_const<P>()),
				p(std::move(p)), weak(0), dynamic_id(dynamic_id), dynamic_ptr(dynamic_ptr)
			{
			}

			std::pair<void*, int> get(cast_graph const& casts, class_id target) const
			{
				// if somebody wants the smart-ptr, he can get a reference to it
				if(target == registered_class<P>::id) return std::pair<void*, int>(&this->p, 0);

				void* naked_ptr = const_cast<void*>(static_cast<void const*>(weak ? weak : get_pointer(p)));
				if(!naked_ptr) return std::pair<void*, int>(nullptr, 0);

				using pointee_type = typename std::remove_cv<typename std::remove_reference<decltype(*get_pointer(p))>::type>::type;

				return casts.cast(naked_ptr,
					registered_class< pointee_type >::id,
					target, dynamic_id, dynamic_ptr);
			}

			explicit operator bool() const
			{
				return p ? true : false;
			}

			void release()
			{
				weak = const_cast<void*>(static_cast<void const*>(get_pointer(p)));
				release_ownership(p);
			}

		private:
			mutable P p;
			// weak will hold a possibly stale pointer to the object owned
			// by p once p has released it's owership. This is a workaround
			// to make adopt() work with virtual function wrapper classes.
			void* weak;
			class_id dynamic_id;
			void* dynamic_ptr;
		};

		template <class ValueType>
		class value_holder :
			public instance_holder
		{
		public:
			// No need for dynamic_id / dynamic_ptr, since we always get the most derived type
			value_holder(lua_State* /*L*/, ValueType val)
				: instance_holder(false), val_(std::move(val))
			{}

			explicit operator bool() const
			{
				return true;
			}

			std::pair<void*, int> get(cast_graph const& casts, class_id target) const
			{
				const auto this_id = registered_class<ValueType>::id;
				void* const naked_ptr = const_cast<void*>((const void*)&val_);
				if(target == this_id) return std::pair<void*, int>(naked_ptr, 0);
				return casts.cast(naked_ptr, this_id, target, this_id, naked_ptr);
			}

			void release() override
			{}

		private:
			ValueType val_;
		};

		/*
			Pointer types should automatically convert to reference types
		*/
		template <class ValueType>
		class pointer_like_holder :
			public instance_holder
		{
		public:
			// No need for dynamic_id / dynamic_ptr, since we always get the most derived type
			pointer_like_holder(lua_State* /*L*/, ValueType val, class_id dynamic_id, void* dynamic_ptr)
				:
				instance_holder(std::is_const< decltype(*get_pointer(val)) >::value),
				val_(std::move(val)),
				dynamic_id_(dynamic_id),
				dynamic_ptr_(dynamic_ptr)
			{
			}

			explicit operator bool() const
			{
				return val_ ? true : false;
			}

			std::pair<void*, int> get(cast_graph const& casts, class_id target) const
			{
				const auto value_id = registered_class<ValueType>::id;
				void* const naked_value_ptr = const_cast<void*>((const void*)&val_);
				if(target == value_id) return std::pair<void*, int>(naked_value_ptr, 0);
				// If we were to support automatic pointer conversion, this would be the place

				using pointee_type = typename std::remove_cv<typename std::remove_reference<decltype(*get_pointer(val_))>::type >::type;
				const auto pointee_id = registered_class< pointee_type >::id;
				void* const naked_pointee_ptr = const_cast<void*>((const void*)get_pointer(val_));
				return casts.cast(naked_pointee_ptr, pointee_id, target, dynamic_id_, dynamic_ptr_);
			}

			void release() override
			{}

		private:
			ValueType val_;
			class_id dynamic_id_;
			void* dynamic_ptr_;
			// weak? must understand what the comment up there really means
		};

	}

} // namespace luabind::detail

#endif // LUABIND_INSTANCE_HOLDER_081024_HPP

