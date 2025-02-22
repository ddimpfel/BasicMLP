# Building the project

## From this directory

Make the app

```sh
# -DBUILD_SHARED_LIBS=OFF sets cmake to use STATIC libraries
cmake -B ./build -DBUILD_SHARED_LIBS=OFF -S ./app
```

Then build the app (for MVSC [--config Debug or Release])

```sh
cmake --build ./build --config Debug
# New build/src directory that the executable is stored in:
cd ./build/src/Debug
basic_mlp
```
