# Define lists of files
set(
        SOURCE_FILES
        gmm.tpp
)
set(
        HEADER_FILES
        gmm.hpp
)
set(
        PYTHON_FILES
        gmm_pybindings.cpp
)
#set(
#        LINK_LIBARARIES
#        proto_plotly_cpp
#        ${Protobuf_LIBRARIES}
#        zmq
#        ${ZMQ_LIBRARIES}
#)


#add_subdirectory(/home/soraxas/git-repo/backward-cpp backward)

add_library(gmm ${SOURCE_FILES} ${HEADER_FILES} ${BACKWARD_ENABLE})
set_target_properties(gmm PROPERTIES LINKER_LANGUAGE CXX)


find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    target_link_libraries(gmm ${OpenMP_CXX_LIBRARIES})
endif ()

#set_property(TARGET gmm PROPERTY POSITION_INDEPENDENT_CODE ON)
##find_package(Backward REQUIRED)

#add_backward(gmm)
#target_compile_options(gmm PRIVATE -fPIC)

#target_link_libraries(gmm  Backward::Backward)

#list(APPEND CMAKE_MODULE_PATH /home/soraxas/git-repo/backward-cpp)
#find_package(Backward REQUIRED)
## This will add libraries, definitions and include directories needed by backward
## through an IMPORTED target.
#target_link_libraries(gmm Backward::Backward)


# external (header-only) library
target_include_directories(gmm PRIVATE
        gcem/include/
        stats/include/
        )

# autodiff package
#target_link_libraries(gmm autodiff)

set_target_properties(gmm
        PROPERTIES
        PUBLIC_HEADER "${HEADER_FILES}"
        )

install(
        TARGETS gmm
        EXPORT ${PROJECT_NAME}Targets
        LIBRARY DESTINATION ${INSTALL_LIB_DIR}
        PUBLIC_HEADER DESTINATION ${INSTALL_INCLUDE_DIR}
        INCLUDES DESTINATION ${INSTALL_INCLUDE_DIR}
)


set(BUILD_TESTER FALSE)
if (${BUILD_TESTER})
#    add_subdirectory(tests)
endif ()


set(BUILD_PYBINDINGS TRUE)

if (${BUILD_PYBINDINGS})

    find_package(pybind11 REQUIRED)

    set(LIB_GMM_DIFF_PY fast_gmm_diff)

    pybind11_add_module(${LIB_GMM_DIFF_PY}
            ${PYTHON_FILES}
            )

    target_link_libraries(${LIB_GMM_DIFF_PY} PRIVATE gmm)

    set_target_properties(${gmm_diff}
            PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON
            PREFIX ""
            )

    install(TARGETS ${LIB_GMM_DIFF_PY}
            EXPORT ${PROJECT_NAME}Targets
            PUBLIC_HEADER DESTINATION ${INSTALL_INCLUDE_DIR}
            INCLUDES DESTINATION ${INSTALL_INCLUDE_DIR}

            COMPONENT python
            LIBRARY DESTINATION "${PYTHON_LIBRARY_DIR}"
            LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
            )

    # Add all targets to the build-tree export set
    export(
            TARGETS ${LIB_GMM_DIFF_PY}
            APPEND FILE "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake"
    )

endif ()

