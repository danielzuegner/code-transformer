Setup
=====

Semantic executable
-------------------

We generate ASTs from code using the (github/semantic)[https://github.com/github/semantic] tool.
One can either use one of the provided docker images or build your own executable by running `cabal v2-build --enable-executable-static` in the repository.
The `--enable-executable-static` flag tells cabal to statically link all necessary libraries into the executable which makes it easier to run the semantic on other machines.
 