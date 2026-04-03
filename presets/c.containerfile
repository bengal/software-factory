FROM software-factory-base

RUN dnf install -y gcc gcc-c++ cmake meson ninja-build pkg-config && dnf clean all
