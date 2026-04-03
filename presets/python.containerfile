FROM software-factory-base

RUN dnf install -y python3-devel gcc && dnf clean all
RUN pip3 install pytest ruff mypy
