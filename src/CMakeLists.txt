cmake_minimum_required(VERSION 3.1)
set (CMAKE_CXX_STANDARD 17)
project( TextDetector )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_library(clustering SHARED IMPORTED)
set_property(TARGET clustering PROPERTY IMPORTED_LOCATION clustering/core/x64/ccore.so)

include_directories( clustering/src )
add_executable( TextDetector TextDetector.cpp )
target_link_libraries( TextDetector ${OpenCV_LIBS} clustering)
set(CMAKE_BUILD_TYPE Debug)

# cmake -DCMAKE_BUILD_TYPE=Debug . for debug
# cmake . for release
