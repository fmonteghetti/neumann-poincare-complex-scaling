#!/bin/sh
# Setup docker environment.
containerWorkspace="$1"
# Upgrade pip (old versions do not support local editable installs)
pip3 install --upgrade pip
# Needed for gmsh
ln -s /usr/bin/python3 ~/.local/bin/python 
cd ${containerWorkspace} && pip3 install -r requirements.txt
# Install python dependencies
pip3 install -e ${containerWorkspace}
# For vscode jupyter extension
pip3 install ipykernel pandas