cmake_minimum_required(VERSION 3.1)
set (CMAKE_CXX_STANDARD 17)
project( TextDetector )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( TextDetector TextDetector.cpp )
target_link_libraries( TextDetector ${OpenCV_LIBS} )

# cmake -DCMAKE_BUILD_TYPE=Debug . for debug
# cmake . for release
