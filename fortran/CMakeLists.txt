cmake_minimum_required(VERSION 3.20)

#[==============================================================================================[
#                                    Basic project definition                                   #
]==============================================================================================]

project(impres 
        DESCRIPTION "Fortran Tuition Material"
		LANGUAGES Fortran)

#[==============================================================================================[
#                                      Main Targets                                            #
]==============================================================================================]

add_library(algorithms_lib)
set_target_properties(algorithms_lib PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
        EXPORT_NAME algorithms
        OUTPUT_NAME algorithms
)

# Source location
add_subdirectory(src)

#[==============================================================================================[
#                                      External Libraries                                       #
]==============================================================================================]

find_package(BLAS REQUIRED)

#[==============================================================================================[
#                                         Tests                                                #
]==============================================================================================]

add_executable(UnitTests)
set_target_properties(UnitTests PROPERTIES
        OUTPUT_NAME unit_tests_exe
)

target_link_libraries(UnitTests PRIVATE algorithms_lib)

# Unit test source location
add_subdirectory(unit_tests)
