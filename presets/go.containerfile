FROM software-factory-base

RUN dnf install -y golang && dnf clean all
