# BZK Address Parsing


## Hardware Requirements

The high computation requirements of the experiments require a GPU or other hyperparallelization hardware compatible with pytorch. The DeepParse library can only take advantage of Nvidia GPUs and will otherwise default to CPU. Remaining experiments should be able to take advantage of any pytorch compatible hardware as long as the correct pytorch binaries are installed.

## Software Dependencies

Every python dependency can be installed using TODO

Aside these dependencies, docker and docker-compose are also required to test the libpostal library.