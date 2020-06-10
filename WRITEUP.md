# Project Write-up

## Edge vs Cloud

	It's a never ending debate on which is better between cloud and edge but there are areas where Edge Aces.
	Edge computing provides following features which makes it efficient as compare to Cloud Computing:

		*  secure and is DDOS free.
		*  provides more time-sensitive and responsive processing.
		*  is less dependent on network connectivity, which increases reliability in the face of unknown connectivity.
		*  enables dedicated processing for a single, specific task.
		*  provides ready access to highly individualized data.





## Explaining Custom Layers

Custom layers are layers that are not included into a list of known layers. If your topology contains any layers that are not in the list of known layers, the Model Optimizer classifies them as custom.

As there was no custom layers in my any model and model optimizer did well.



## Converting a tf Model :point_left:
Since I've used tensorflow model and for that I've used OpenVINO Toolkit Model Optimizer
	To convert a TensorFlow model:

		1. Go to the <INSTALL_DIR>/deployment_tools/model_optimizer directory
		2. Use the mo_tf.py script to simply convert a model with the path to the input model .pb file:
		3. '''<python3 mo_tf.py --input_model <INPUT_MODEL>.pb>'''

	Two groups of parameters are available to convert your model:

		*Framework-agnostic parameters: These parameters are used to convert any model trained in any supported framework.
		*TensorFlow-specific parameters: Parameters used to convert only TensorFlow models.
	

	 While other models were converted using model optimizer via code '''python3 mo.py --input_model squeezenet.caffemodel --input_proto squeezenet.prototxt''' 

_Before running the tool with a trained model, make sure the model is converted to the Inference Engine format (*.xml + *.bin) using the Model Optimizer tool._		




## Comparing Model Performance


Note: Model inference comparison and remarks were done on the workspace while for other performance I've used my laptop to compare.


Laptop's Config:
	Processor(s):1 Processor(s) Installed[ Intel64 Family 6 Model 78 Stepping 3 GenuineIntel ~2000 Mhz]
	RAM: 8 GB (DDR-4)


| Model Name                          | Inference Time | Throughput | Probability Threshhold | Count(iterations) | Duration   | Latency   | Remarks                                            
--------------------------------------|----------------|------------|------------------------|-------------------|------------|-----------|----------------------------------------------------
| SSD inception_V2(tf)                | 156            | 11.35 FPS  | 0.7                    | 688               | 60604 ms   | 349.49 ms | misses detection,multiple detection of same object 
| pedestrian-detection-adas-0002 FP16 | 56             | 24.74      | 0.85                   | 1488              | 6012.89 ms | 151.04 ms | misses detection                                   
| pedestrian-detection-adas-0002 FP16 | 53             | NA         | NA                     | NA                | NA         | NA        | showed detection while there was no object.        


1. I started with ssd mobilenet v1 model and ended up with very poor bounding box detection and it's size. :no_mouth: 
2. ssd mobilenet v2 was also used and it gave flase/missed detection of object. :neutral_face:
3. SSD inception_V2(tf) model was 97.3 MB pre-conversion and 95.4 MB post-conversion. :expressionless:

4. I chose pedestrian-detection-adas-0002 FP16 from open-model Zoo by Intel and it showed too much missing of boxes. :relieved:


 _Final model choosen was pedestrian-detection-adas-0002 FP16_




**My method(s) to compare models before and after conversion to Intermediate Representations
were:**

1. To use infernece time as one of the parameter. 

2. checking over low light vs good light
	   **Pre-conversion**

	* SSD model missed many detections in ambient light while in good light missed few and detected one object into multiple in single frame.

  	   **post-conversion**
	* SSD model missed detection/multiple-detection in single frame while object was in frame also shown detection while object was not in frame  in ambient 
	* In Good light missed few and detected one object into multiple in single frame.


	* with pedestrian-detection-adas-0002 FP16 performed quite better than SSD.

	* shown good performance in ambient as well as good light although there were chances when it detected false fewer in ambient light.
  

3. used benchmark_tool provided by Intel OpenVINO Toolkit over my laptop by feeding single image asynchronously and checking following parameters:
	*Throughput
	*Count
	*Latency
	*Duration

## Assess Model Use Cases

1. Can be used at retail stores or marts to count. (Security Reasons)
2. To check tht no more than particular no. of people enters restricted place.( Safety Purposes)
3. Counting no. of person go to store and how average they spend time shopping in a single store.(help to open shop & close shops and manage workers at peak times)


## Assess Effects on End User Needs

1. Make sure the deployed device is provided with good amount of light
2. Camera angle should be t particular distance like abt 3 feet or so above body.
3. Image Size probably be approx 300x300 for better transmission of visuals and stats over internet.

  























