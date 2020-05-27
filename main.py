
"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
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


# MQTT server environment variables

import os
import sys
import time
import socket
import json
import cv2
import logging as log
import paho.mqtt.client as mqtt
from argparse import ArgumentParser
from inference import Network
CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
"sofa", "train", "tvmonitor"]
IGNORE = set(["background", "aeroplane", "bicycle", "bird", "boat",
"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
"dog", "horse", "motorbike", "pottedplant", "sheep",
"sofa", "train", "tvmonitor"])


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser=ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    parser.add_argument("-pc", "--perf_counts", type=str, default=False,
                        help="Print performance counters")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client=mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client
'''def performance_counts(perf_count):
      
    #priwnt information about layers of the model.
    
    print("{:<70} {:<15} {:<15} {:<15} {:<10}".format('name', 'layer_type','exec_type', 'status','real_time, us'))
    for layer, stats in perf_count.items():
        print("{:<70} {:<15} {:<15} {:<15} {:<10}".format(layer,
                                                          stats['layer_type'],
                                                          stats['exec_type'],
                                                          stats['status'],
                                                          stats['real_time']))'''


def infer_on_stream(args, client,k):
    
    print("Hello")
    # Checks for live feed
    

    
    

    # Enter video file
    input_stream = args.input
    
    cap = cv2.VideoCapture(input_stream)

    if input_stream:
        cap.open(args.input)

    if not cap.isOpened():
        log.error("ERROR! Unable to open video source")
    global width, height, prob_threshold
    prob_threshold = args.prob_threshold
    width = cap.get(3)
    height = cap.get(4)
    while cap.isOpened():
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)


    cur_request_id=0
    last_count=0
    total_count=0
    start_time=0
    # Initialise the class
    plugin=Network()
    
    
    ### TODO: Load the model through `infer_network` ###
    plugin.load_model(args.model, cur_request_id, args.device, CPU_EXTENSION)
    
    net_input_shape=plugin.get_input_shape()
    ### TODO: Handle the input stream ###

    cap.open(args.input)
    width=int(cap.get(3))
    height=int(cap.get(4))
    
    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        # Read the next frame
        ### TODO: Read from the video capture ###
        flag, frame=cap.read()
        if not flag:
            break
        key_pressed=cv2.waitKey(60)


    

        

        ### TODO: Pre-process the image as needed ###
        p_frame=cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame=p_frame.transpose((2, 0, 1))
        p_frame=p_frame.reshape(1, *p_frame.shape)
        inf_start=time.time()
        ### TODO: Start asynchronous inference for specified request ###
        plugin.exec_net(p_frame)
        ### TODO: Wait for the result ###

            ### TODO: Get the results of the inference request ###
        if plugin.wait(cur_request_id) == 0:
         
         
            det_time=time.time() - inf_start
            # Results of the output layer of the network
            result=plugin.get_output(cur_request_id)
            
            #performace checking


            
            # TODO: Update the frame to include detected bounding boxes
            
         curr_count=0
         for obj in result[0][0]:
         
         # Draw bounding box for object when it's probability is more than
         # the specified threshold
          f=str(obj[2]*100)+'%'
        
        
          if obj[2] > prob_threshold:
           x_min=int(obj[3] * width)
           y_min=int(obj[4] * height)
           x_max=int(obj[5] * width)
           y_max=int(obj[6] * height)
           # Write out the frame
           cv2.rectangle(frame, (x_min, y_min),(x_max, y_max), (0, 55, 255), 1)
           cv2.putText(frame, 'score'+f, (15, 15),cv2.FONT_HERSHEY_COMPLEX, 0.5, (150, 0, 0),1)
           curr_count=curr_count + 1         
            
            
            
            
            
            
            
            inf_time_message = "Inference time: {:.3f}ms"\
                               .format(det_time * 1000)
            cv2.putText(frame, inf_time_message, (15,45),cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
            
            # When new person enters the video
            duration=0
            if curr_count > last_count:
                start_time=time.time()
                total_count=total_count + (curr_count - last_count)
                client.publish("person", json.dumps({"total": total_count}))

            # Person duration in the video is calculated
            if curr_count < last_count:
                duration=int(time.time() - start_time)
                # Publish messages to the MQTT server
                ### TODO: Calculate and send relevant information on ###
                client.publish("person/duration",json.dumps({"duration": duration}))
            client.publish("person", json.dumps({"count": curr_count}))
            last_count=curr_count
            
            if key_pressed == 27:
                break

            

           
           ### TODO: Send the frame to the FFMPEG server ### 
            sys.stdout.buffer.write(frame)
            sys.stdout.flush()


            ### Topic "person/duration": key of "duration" ###

 

        ### TODO: Write an output image if `single_image_mode` ###
            #if single_image_mode:
            #cv2.imwrite('output_image.jpg', frame)
    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()
    


def main():
    


    """
    Load the network and parse the output.

    :return: None
    """
    global width, height
    # Grab command line args
    args=build_argparser().parse_args()
    # Connect to the MQTT server
    client=connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client,k)
    return


if __name__ == '__main__':
    main()

# python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m intel/pedestrian-detection-adas-0002/INT8/pedestrian-detection-adas-0002.xml  -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.7  | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

#
