# Creating a CMake app

## Configuring and Adding Dependencies

Configure the name of your project by setting the CMake variable
PROJECT_NAME in the root directory CMake script:

```cmake
set(PROJECT_NAME project_name)
```

This is a demo app to create a CMake files using git submodules:

```sh
git submodule add GIT_URL.git ./vendor/REPO_NAME
```

In order to use the repositories in a CMake project (easily),
they must have CMake scripts already included. If this is the case,
add them to ./vendor/CMakeLists.txt as a subdirectory:

```cmake
add_subdirectory(sfml)
add_subdirectory(imgui-sfml)
```

Create a separate build directory at the same level as the app directory:

```sh
mkdir ../build
```

## Building the Project

The reasoning for creating a separate build directory is to keep all of
the gory build details out of the main git repository and make refreshing
of the build easier.

Navigate into the build directory (last step created) and run the cmake
generator on the app source directory:

```sh
# -DBUILD_SHARED_LIBS=OFF sets cmake to use STATIC libraries
cmake -DBUILD_SHARED_LIBS=OFF ../app
```

Then build and run the project (for MVSC [--config Debug or Release]):

```sh
cmake --build . --config Debug
# this is a new build/src directory that the executable is stored in
./src/Debug/cmake_app_exe.exe
```
