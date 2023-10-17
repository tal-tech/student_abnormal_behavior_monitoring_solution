#/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../3rdParty/ubuntu-cuda10
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64
echo $LD_LIBRARY_PATH
./GodeyeVideoCrow