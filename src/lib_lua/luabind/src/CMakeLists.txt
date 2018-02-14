# Build for LuaBind
# Ryan Pavlik <rpavlik@iastate.edu>
# http://academic.cleardefinition.com/
# Iowa State University HCI Graduate Program/VRAC

# ls *.cpp | sort | sed "s/^/    /"
set(LUABIND_SRCS
    class.cpp
    class_info.cpp
    class_registry.cpp
    class_rep.cpp
    create_class.cpp
    error.cpp
    exception_handler.cpp
    function.cpp
    function_introspection.cpp
    inheritance.cpp
    link_compatibility.cpp
    lua51compat.cpp
    object_rep.cpp
    open.cpp
    operator.cpp
    pcall.cpp
    scope.cpp
    set_package_preload.cpp
    shared_ptr_converter.cpp
    stack_content_by_name.cpp
    weak_ref.cpp
    wrapper_base.cpp)

# ls ../luabind/*.hpp | sort | sed "s/^/    /"
set(LUABIND_API
    ../luabind/adopt_policy.hpp
    ../luabind/back_reference_fwd.hpp
    ../luabind/back_reference.hpp
    ../luabind/class.hpp
    ../luabind/class_info.hpp
    ../luabind/config.hpp
    ../luabind/container_policy.hpp
    ../luabind/copy_policy.hpp
    ../luabind/dependency_policy.hpp
    ../luabind/discard_result_policy.hpp
    ../luabind/error_callback_fun.hpp
    ../luabind/error.hpp
    ../luabind/exception_handler.hpp
    ../luabind/from_stack.hpp
    ../luabind/function_converter.hpp
    ../luabind/function.hpp
    ../luabind/function_introspection.hpp
    ../luabind/get_main_thread.hpp
    ../luabind/get_pointer.hpp
    ../luabind/handle.hpp
    ../luabind/intrusive_ptr_converter.hpp
    ../luabind/iterator_policy.hpp
    ../luabind/luabind.hpp
    ../luabind/lua_include.hpp
    ../luabind/lua_state_fwd.hpp
    ../luabind/make_function.hpp
    ../luabind/nil.hpp
    ../luabind/no_dependency.hpp
    ../luabind/object_fwd.hpp
    ../luabind/object.hpp
    ../luabind/open.hpp
    ../luabind/operator.hpp
    ../luabind/out_value_policy.hpp
    ../luabind/prefix.hpp
    ../luabind/raw_policy.hpp
    ../luabind/return_reference_to_policy.hpp
    ../luabind/scope.hpp
    ../luabind/set_package_preload.hpp
    ../luabind/shared_ptr_converter.hpp
    ../luabind/stack.hpp
    ../luabind/std_shared_ptr_converter.hpp
    ../luabind/tag_function.hpp
    ../luabind/typeid.hpp
    ../luabind/value_wrapper.hpp
    ../luabind/version.hpp
    ../luabind/weak_ref.hpp
    ../luabind/wrapper_base.hpp
    ../luabind/yield_policy.hpp)
source_group(API FILES ${LUABIND_API})

# ls ../luabind/detail/*.hpp | sort | sed "s/^/    /"
set(LUABIND_DETAIL_API
    ../luabind/detail/call_function.hpp
    ../luabind/detail/call.hpp
    ../luabind/detail/call_member.hpp
    ../luabind/detail/call_operator_iterate.hpp
    ../luabind/detail/class_registry.hpp
    ../luabind/detail/class_rep.hpp
    ../luabind/detail/constructor.hpp
    ../luabind/detail/convert_to_lua.hpp
    ../luabind/detail/debug.hpp
    ../luabind/detail/decorate_type.hpp
    ../luabind/detail/deduce_signature.hpp
    ../luabind/detail/enum_maker.hpp
    ../luabind/detail/format_signature.hpp
    ../luabind/detail/garbage_collector.hpp
    ../luabind/detail/has_get_pointer.hpp
    ../luabind/detail/inheritance.hpp
    ../luabind/detail/instance_holder.hpp
    ../luabind/detail/link_compatibility.hpp
    ../luabind/detail/make_instance.hpp
    ../luabind/detail/most_derived.hpp
    ../luabind/detail/object_call.hpp
    ../luabind/detail/object.hpp
    ../luabind/detail/object_rep.hpp
    ../luabind/detail/operator_id.hpp
    ../luabind/detail/other.hpp
    ../luabind/detail/pcall.hpp
    ../luabind/detail/pointee_sizeof.hpp
    ../luabind/detail/policy.hpp
    ../luabind/detail/primitives.hpp
    ../luabind/detail/property.hpp
    ../luabind/detail/signature_match.hpp
    ../luabind/detail/stack_utils.hpp
    ../luabind/detail/typetraits.hpp
    ../luabind/detail/yes_no.hpp)
source_group(Internal FILES ${LUABIND_DETAIL_API})

set(LUABIND_EXTRA_LIBS)
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4251")
elseif(CMAKE_COMPILER_IS_GNUCXX)
    # TODO we often need libm, not just with gcc.
    list(APPEND LUABIND_EXTRA_LIBS m)
endif()

set(CMAKE_DEBUG_POSTFIX "-d")
add_library(luabind ${LUABIND_SRCS} ${LUABIND_API} ${LUABIND_DETAIL_API})
target_link_libraries(luabind ${LUA_LIBRARIES} ${LUABIND_EXTRA_LIBS} ${CMAKE_DL_LIBS})
set_property(TARGET
    luabind PROPERTY
    COMPILE_FLAGS
    ${CMAKE_SHARED_LIBRARY_C_FLAGS})

set_property(TARGET
    luabind
    PROPERTY
    PROJECT_LABEL
    "LuaBind Library")

if (LUABIND_APPEND_VERSION_SUFFIX)
    set_property(TARGET
        luabind
        PROPERTY
        OUTPUT_NAME
        "luabind09")
else()
    # This is neccesary when the LUABIND_APPEND_VERSION_SUFFIX variable is
    # changed from ON to OFF.
    set_property(TARGET
        luabind
        PROPERTY
        OUTPUT_NAME
        "luabind")
endif()

set(ALLHEADERS ${LUABIND_API} ${LUABIND_DETAIL_API} PARENT_SCOPE)
set(APIHEADERS ${LUABIND_API} PARENT_SCOPE)

if(LUABIND_INSTALL)
    if(NOT INCLUDE_DIR)
        set(INCLUDE_DIR include)
    endif()
    if(NOT LIB_DIR)
        set(LIB_DIR lib)
    endif()
    if(NOT ARCH_DIR)
        set(ARCH_DIR lib)
    endif()
    if(NOT BIN_DIR)
        set(BIN_DIR bin)
    endif()
    install(FILES ${LUABIND_API}
        DESTINATION ${INCLUDE_DIR}/luabind
        COMPONENT sdk)
    install(FILES ${LUABIND_DETAIL_API}
        DESTINATION ${INCLUDE_DIR}/luabind/detail
        COMPONENT sdk)
    install(TARGETS luabind
        EXPORT luabind
        RUNTIME DESTINATION ${BIN_DIR} COMPONENT runtime
        LIBRARY DESTINATION ${LIB_DIR} COMPONENT runtime
        ARCHIVE DESTINATION ${ARCH_DIR} COMPONENT sdk)
endif()
