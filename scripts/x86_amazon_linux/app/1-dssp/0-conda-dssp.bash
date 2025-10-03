conda install -c salilab dssp

# check which boost libraries to install
ldd "$(which mkdssp)" | grep -i boost
