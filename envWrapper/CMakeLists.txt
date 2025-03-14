cmake_minimum_required(VERSION 3.16)
project(envWrapper)

# Set C++ version
set(CMAKE_CXX_STANDARD 17)

IF(NOT CMAKE_BUILD_TYPE)
    #SET(CMAKE_BUILD_TYPE "DEBUG")
    SET(CMAKE_BUILD_TYPE "RELEASE")
    #SET(CMAKE_BUILD_TYPE "RELWITHDEBINFO")
    #SET(CMAKE_BUILD_TYPE "MINSIZEREL")
ENDIF()

include_directories("../inc")
include_directories("../default_planner")
include_directories("./")

# Add LRR environment files excluding driver.cpp with main fucntion
file(GLOB SOURCES "../src/*.cpp" "../default_planner/*.cpp" "../my_scheduler/*.cpp" "./*.cpp")
list(FILTER SOURCES EXCLUDE REGEX ".*driver\\.cpp$")

#my stuff
include_directories("../my_scheduler")
# include_directories("../third_party_libraries/or-22/or-tools/include")
# link_directories("../third_party_libraries/or-22/or-tools/lib")
execute_process(COMMAND lsb_release -sr
    OUTPUT_VARIABLE UBUNTU_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(UBUNTU_VERSION VERSION_GREATER_EQUAL 24)
    message(STATUS "Running on Ubuntu 24 or later")
    include_directories("../third_party_libraries/or-24/or-tools/include")
    link_directories("../third_party_libraries/or-24/or-tools/lib")
elseif(UBUNTU_VERSION VERSION_GREATER_EQUAL 22)
    message(STATUS "Running on Ubuntu 22")
    include_directories("../third_party_libraries/or-22/or-tools/include")
    link_directories("../third_party_libraries/or-22/or-tools/lib")
else()
    message(STATUS "Unknown Ubuntu version: ${UBUNTU_VERSION}")
endif()

# Find Boost
ADD_DEFINITIONS(-DBOOST_LOG_DYN_LINK)
#SET(Boost_USE_STATIC_LIBS ON)
find_package( Boost 1.49.0 REQUIRED COMPONENTS program_options system filesystem log log_setup)

include_directories( ${Boost_INCLUDE_DIRS} )

# Find Python and pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

# Add source files
add_library(${PROJECT_NAME} MODULE envWrapper.cpp ${SOURCES})

# Link libraries (pybind11 and other dependencies)
target_link_libraries(envWrapper PRIVATE pybind11::module ${Boost_LIBRARIES} ortools absl_base absl_strings)

# Output the shared library with a Python-compatible name
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "" SUFFIX ".so")
