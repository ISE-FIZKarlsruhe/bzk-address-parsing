# Use ubuntu for main image, installing C libraries on top of other images is often more complicated
FROM ubuntu:22.04 


# Install build dependencies
RUN apt update && apt install -y curl build-essential autoconf automake libtool pkg-config git wget python3 python3-pip

# Install libpostal and python bindings
RUN cd / && git clone https://github.com/openvenues/libpostal && \
    cd libpostal && \
    ./bootstrap.sh && \
    ./configure --datadir=/data/libpostal && \
    make && \
    make install && \
    ldconfig
RUN pip install postal bottle

WORKDIR /workspace
COPY libpostal_server.py /workspace/libpostal_server.py
CMD ["python3", "libpostal_server.py"]