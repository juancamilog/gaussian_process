CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
PROJECT(gaussian_process)
 
SET (CMAKE_MODULE_PATH "/usr/share/cmake/Modules")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

SET(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=gnu++0x")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse -msse2 -msse3 -O3 -DNDEBUG")

SET( EIGEN3_INCLUDE_DIR  ${PROJECT_SOURCE_DIR}/include/eigen )
#SET( DLIB_SOURCE_DIR  ${PROJECT_SOURCE_DIR}/include/dlib-18.4 )
SET( ALGLIB_SOURCE_DIR  ${PROJECT_SOURCE_DIR}/include/alglib/src )

#INCLUDE_DIRECTORIES( ${PROJECT_SOURCE_DIR}/include ${EIGEN3_INCLUDE_DIR} ${DLIB_SOURCE_DIR})
INCLUDE_DIRECTORIES( ${PROJECT_SOURCE_DIR}/include ${EIGEN3_INCLUDE_DIR} ${ALGLIB_SOURCE_DIR})

#alglib
add_library(alglib SHARED
${ALGLIB_SOURCE_DIR}/ap.cpp
${ALGLIB_SOURCE_DIR}/alglibinternal.cpp
${ALGLIB_SOURCE_DIR}/alglibmisc.cpp
${ALGLIB_SOURCE_DIR}/dataanalysis.cpp
${ALGLIB_SOURCE_DIR}/diffequations.cpp
${ALGLIB_SOURCE_DIR}/fasttransforms.cpp
${ALGLIB_SOURCE_DIR}/integration.cpp
${ALGLIB_SOURCE_DIR}/interpolation.cpp
${ALGLIB_SOURCE_DIR}/linalg.cpp
${ALGLIB_SOURCE_DIR}/optimization.cpp 
${ALGLIB_SOURCE_DIR}/solvers.cpp
${ALGLIB_SOURCE_DIR}/specialfunctions.cpp
${ALGLIB_SOURCE_DIR}/statistics.cpp)

# General
ADD_LIBRARY(gaussian_process SHARED ${PROJECT_SOURCE_DIR}/src/gaussian_process.cpp)
TARGET_LINK_LIBRARIES(gaussian_process alglib)
ADD_EXECUTABLE(testGaussianProcess ${PROJECT_SOURCE_DIR}/src/testGaussianProcess.cpp)
TARGET_LINK_LIBRARIES(testGaussianProcess gaussian_process)
#ADD_EXECUTABLE(testEigen ${PROJECT_SOURCE_DIR}/src/testEigen.cpp)
#ADD_EXECUTABLE(testAlglib ${PROJECT_SOURCE_DIR}/src/testAlglib.cpp)
#TARGET_LINK_LIBRARIES(testAlglib alglib)
