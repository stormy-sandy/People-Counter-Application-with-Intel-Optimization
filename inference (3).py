#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None
    def load_model(self, model, num_req,device="CPU" ,cpu_extension=None):
                
        ### TODO: Load the model ###
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        ### TODO: Check for supported layers ###
        
        
        
        
        
        ### TODO: Add any necessary extensions ###
        ### TODO: Return the loaded inference plugin ###
        self.plugin = IECore()

        # Add a CPU extension, if applicable
        if cpu_extension and "CPU" in device:
            self.plugin.add_extension(cpu_extension, device)
         # Read the IR as a IENetwork
        self.network = IENetwork(model=model_xml, weights=model_bin)  
        
        # Load the IENetwork into the plugin
        if num_req == 0:
            # Loads network read from IR to the plugin
            self.exec_network = self.plugin.load_network(self.network, device)
        else:
            self.exec_network = self.plugin.load_network(self.network, device,num_req=num_req)

       

        
        
        
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))
        

       
        return self.plugin
        ### Note: You may need to update the function parameters. ###
        

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        return self.network.inputs[self.input_blob].shape
    
    def performance_counter(self, request_id):
        #performance counter
        perf_count = self.exec_network.requests[request_id].get_perf_counts()
        return perf_count

    def exec_net(self,image):
        ### TODO: Start an asynchronous request ###
        self.exec_network.start_async(request_id=0,inputs={self.input_blob: image})
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return self.plugin

    def wait(self,request_id):
        ### TODO: Wait for the request to be complete. ###
        status = self.exec_network.requests[request_id].wait(-1)
        
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return status

    def get_output(self,request_id,output=None):
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        if output:
            res = self.infer_request_handle.outputs[output]
        else:
            res = self.exec_network.requests[request_id].outputs[self.output_blob]
        return res
        
