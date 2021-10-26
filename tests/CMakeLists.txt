
add_executable(main main.cpp)
target_link_libraries(main gmm)

target_include_directories(
        main
        PRIVATE ${CMAKE_SOURCE_DIR}/src/GaussianMixtureModel
)


#add_subdirectory(/home/soraxas/git-repo/backward-cpp backward)
#
##set_property(TARGET gmm PROPERTY POSITION_INDEPENDENT_CODE ON)
#find_package(Backward)
#add_backward(main)
