#!/bin/bash

sudo cp /home/laibah/raynet/build/omnetbind.cpython-311-x86_64-linux-gnu.so /usr/local/lib
sudo cp /home/laibah/raynet/build/librlomnet.so /usr/local/lib
sudo cp /home/laibah/raynet/build/libCmdrlenv.so /usr/local/lib
sudo cp /home/laibah/raynet/simlibs/RLComponents/src/libRLComponents.so /usr/local/lib
sudo cp /home/laibah/raynet/simlibs/RLCC/src/libRLCC.so /usr/local/lib
sudo cp /home/laibah/inet4.5/src/libINET.so /usr/local/lib

ldconfig -n /usr/local/lib