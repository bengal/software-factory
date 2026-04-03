FROM software-factory-base

RUN dnf install -y gcc gcc-c++ iproute dnsmasq && dnf clean all

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"
