FROM mcr.microsoft.com/vscode/devcontainers/miniconda:3

RUN conda install -y python=3.10.13 \
    && pip install --no-cache-dir pipx \
    && pip uninstall -y pipx \
    && pip install pipx

WORKDIR /workspaces/MedApp-Dev-AI

#RUN conda env create -f environment.yml