
# Install Python Dependencies
pip install -r requirement.txt

# System packages
apt-get update
apt install -y python3-dev cmake git

# Conda packages
conda install -c conda-forge swig=4.3.0 gflags -y
conda install mkl mkl-devel -y