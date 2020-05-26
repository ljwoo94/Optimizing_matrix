# Homework 1

## Summary

You should implement sum reduce for parallelization

Go to `reduce.cc` and implement `TODO`

`anwser` and `sum` should be equal

```
$ make submit
$ cat io2.out
init took 0.79228 sec
serial reduce took 0.283715 sec
answer: 576460751766552576
parallel reduce took 0.12185 sec
sum: 576460751766552576
```

## Functions

```
# Run locally
make
make run
# Compile
make reduce
# Clean
make clean
# Submit job to condor
make submit
# Check queue of condor
make queue
# Check status of condor
make remove
```

## References

- [std::thread - cplusplus.com](http://www.cplusplus.com/reference/thread/thread/)
- [OpenMP 4.5 C/C++ Syntax Guide](https://www.openmp.org/wp-content/uploads/OpenMP-4.5-1115-CPP-web.pdf)
- [HTCondor commands cheat-sheet](https://raggleton.github.io/condor-cheatsheet/)
- [CMake Cheatsheet](https://github.com/mortennobel/CMake-Cheatsheet/blob/master/CMake_Cheatsheet.pdf)