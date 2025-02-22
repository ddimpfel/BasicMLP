# Building the tests

## From this directory

Make the tests

```sh
# Return to root to make path tracing easier
cd ../../..
# -DBUILD_SHARED_LIBS=OFF sets cmake to use STATIC libraries
cmake -B ./build/Tests -DBUILD_SHARED_LIBS=OFF -S ./app/src/Tests
```

Then build the tests (for MVSC [--config Debug or Release])

```sh
cmake --build ./build/Tests --config Debug
# New build/src directory that the executable is stored in:
cd ./build/Tests/Debug
tests
```

## Investigate this test command: [cmake.org](https://cmake.org/cmake/help/latest/command/create_test_sourcelist.html)

create_test_sourcelist(\<sourceListName> \<driverName> \<test>... \<options>...)

It builds all tests from source so that they are stored in a single executable. Tests must be called from a function that has the same name as the file.

Further reading: [Testing with CMake](https://cmake.org/cmake/help/book/mastering-cmake/chapter/Testing%20With%20CMake%20and%20CTest.html)
