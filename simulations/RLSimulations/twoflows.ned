//
// Copyright (C) 2004 OpenSim Ltd.
//
// SPDX-License-Identifier: LGPL-3.0-or-later
//


@namespace(inet);
@namespace(RLCC);
@namespace(RLComponents);

package simulations.RLSimulations;

import inet.common.misc.ThruputMeteringChannel;
import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;
import inet.node.inet.Router;
import inet.node.inet.StandardHost;
import RLComponents.Broker;
import RLComponents.Stepper;



network twoflows
{
    types:

        channel ethernetline extends ThruputMeteringChannel
        {
            delay = 10 ms; 
            datarate = 96 Mbps; 
        }
        
        channel fiberline extends ThruputMeteringChannel
        {
            delay = 0ms;
            datarate = 1000Gbps;
        }

    submodules:
        configurator: Ipv4NetworkConfigurator {
            parameters:
                @display("p=100,100;is=s");
        }
        r1: Router {
            @display("p=400,250");
        }
        r2: Router {
            @display("p=577,250");
        }
        cli0: StandardHost {
            parameters:
                @display("i=device/laptop;p=250,100,col,100");
                
        }
        cli1: StandardHost {
            parameters:
                @display("i=device/laptop;p=250,100,col,100");
                
        }
        srv0: StandardHost {
            parameters:
                @display("i=device/server;p=850,250");
        }

        broker: Broker{}
        stepper: Stepper{}
    connections:
        cli0.pppg++ <--> fiberline <--> r1.pppg++;
        cli1.pppg++ <--> fiberline <--> r1.pppg++;
        srv0.pppg++ <--> fiberline <--> r2.pppg++;

        r2.pppg++ <--> ethernetline <--> r1.pppg++;
}

