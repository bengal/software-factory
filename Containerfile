# software-factory base image.
#
# Contains: python3, git, the factory CLI. No language-specific toolchain.
# Extend this for your project via presets or a custom Containerfile.factory.
#
# Build:
#   podman build -t software-factory-base .

FROM fedora:43

RUN dnf install -y \
        git \
        python3 \
        python3-pip \
        make \
        && dnf clean all

COPY . /opt/software-factory
RUN pip3 install /opt/software-factory

RUN git config --global user.name "Software Factory" \
    && git config --global user.email "factory@localhost"

WORKDIR /workspace

ENTRYPOINT ["factory", "run"]
