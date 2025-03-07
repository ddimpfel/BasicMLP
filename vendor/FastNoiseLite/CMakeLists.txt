# Creates interface library
add_library(FastNoiseLite INTERFACE)

# Specify the include dir
target_include_directories(FastNoiseLite INTERFACE
	${CMAKE_CURRENT_SOURCE_DIR}
)