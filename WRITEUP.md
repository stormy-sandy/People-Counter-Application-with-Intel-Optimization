
# Project Write-up

To execute the app run the following command after initiating the servers:

'''

python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m ssd_inception_v2_coco_2018_01_28/frozen_inference_graph.xml  -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.85  | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

'''




***

## Edge vs Cloud

 It's a never ending debate on which is better between cloud and edge but there are areas where Edge Aces.
 Edge computing provides following features which makes it efficient as compare to Cloud Computing:

  * secure and is DDOS free.
  * provides more time-sensitive and responsive processing.
  * is less dependent on network connectivity, which increases reliability in the face of unknown connectivity.
  * enables dedicated processing for a single, specific task.
    * provides ready access to highly individualized data.

***

## Custom Layers

Step 1: Generate: Use the Model Extension Generator to generate the Custom Layer Template Files.
		The Model Extension Generator is included in the Intel® Distribution of OpenVINO™ toolkit installation and is run using the command (here with the "--help" option):
		'python3 <INSTALL_DIR>/deployment_tools/tools/extension_generator/extgen.py new --mo-tf-ext'

Step 2: Edit: Edit the Custom Layer Template Files as necessary to create the specialized Custom Layer Extension Source Code.

Step 3: Specify: Specify the custom layer extension locations to be used by the Model Optimizer or Inference Engine.

>You have two options for TensorFlow* models with custom layers:

1. Register those layers as extensions to the Model Optimizer. In this case, the Model Optimizer generates a valid and optimized Intermediate Representation.
2. If you have sub-graphs that should not be expressed with the analogous sub-graph in the Intermediate Representation, but another sub-graph should appear in the model, the Model Optimizer provides such an option. This feature is helpful for many TensorFlow models. To read more, see [Sub-graph Replacement in the Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_customize_model_optimizer_Subgraph_Replacement_Model_Optimizer.html).

The model I chose for the app was in (supported Frozen Topologies from TensorFlow Object Detection Models Zoo)[https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html#supported_tensorflow_layers] i.e. Supported Topologies,
so there were no unsupported custom layers in model and model optimizer optimized the model without producing any error.

***

## Converting a Model to Intermediate Representation (IR) :point_left:

### Configuring the Model Optimizer:

You must configure the Model Optimizer for the framework that was used to train the model.

To configure a framework, go to the <INSTALL_DIR>/deployment_tools/model_optimizer/install_prerequisites directory and run:
 'install_prerequisites_tf.bat'

### Loading Non-Frozen Models to the Model Optimizer

There are three ways to store non-frozen TensorFlow models and load them to the Model Optimizer:

1. Checkpoint:

* In this case, a model consists of two files:

⋅⋅1. inference_graph.pb or inference_graph.pbtxt
⋅⋅2. checkpoint_file.ckpt
⋅⋅3. If you do not have an inference graph file, refer to Freezing Custom Models in Python.

* To convert such TensorFlow model:

⋅⋅1. Go to the <INSTALL_DIR>/deployment_tools/model_optimizer directory
⋅⋅2. Run the mo_tf.py script with the path to the checkpoint file to convert a model:
    If input model is in .pb format:
		'python3 mo_tf.py --input_model <INFERENCE_GRAPH>.pb --input_checkpoint <INPUT_CHECKPOINT>'
	If input model is in .pbtxt format:
		'python3 mo_tf.py --input_model <INFERENCE_GRAPH>.pbtxt --input_checkpoint <INPUT_CHECKPOINT> --input_model_is_text'

2. MetaGraph:

In this case, a model consists of three or four files stored in the same directory:

	 * model_name.meta
	 * model_name.index
	 * model_name.data-00000-of-00001 (digit part may vary)
	 * checkpoint (optional)
* To convert such TensorFlow model:

⋅⋅1. Go to the '<INSTALL_DIR>/deployment_tools/model_optimizer' directory
Run the 'mo_tf.py' script with a path to the MetaGraph .meta file to convert a model:

	* 'python3 mo_tf.py --input_meta_graph <INPUT_META_GRAPH>.meta'

3. SavedModel:

In this case, a model consists of a special directory with a .pb file and several subfolders: variables, assets, and assets.extra. For more information about the SavedModel directory, refer to the README file in the TensorFlow repository.

To convert such TensorFlow model:

Go to the <INSTALL_DIR>/deployment_tools/model_optimizer directory
⋅⋅1. Run the 'mo_tf.py' script with a path to the SavedModel directory to convert a model:
⋅⋅2. 'python3 mo_tf.py --saved_model_dir <SAVED_MODEL_DIRECTORY>'




>To convert SSD inception V3(tf) model first download it and extract from (link)[http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz]

To convert go to you model directory and execute the following code:

```
python <INSTALL_DIR>/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config <INSTALL_DIR>/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
```

***

## Comparing Model Performance


Note: Model inference comparison and remarks were done on the workspace while for other performance I've used my laptop to compare.


Laptop's Config:
	Processor(s):1 Processor(s) Installed[ Intel64 Family 6 Model 78 Stepping 3 GenuineIntel ~2000 Mhz]
	RAM: 8 GB (DDR-4)


| Model Name                          | Inference Time | Throughput | Probability Threshhold | Count(iterations) | Duration   | Latency   | Remarks
--------------------------------------|----------------|------------|------------------------|-------------------|------------|-----------|----------------------------------------------------
| SSD inception_V2(tf)                | 156            | 11.35 FPS  | 0.7                    | 688               | 60604 ms   | 349.49 ms

1. I started with ssd mobilenet v1 model and ended up with very poor bounding box detection and it's size. :no_mouth:
2. SSD inception_V2(tf) model was 97.3 MB pre-conversion and 95.4 MB post-conversion.

**My method(s) to compare models before and after conversion to Intermediate Representations
were:**

3. used benchmark_tool provided by Intel OpenVINO Toolkit over my laptop by feeding single image asynchronously and checking following parameters:
	* Throughput
	* Count
	* Latency
	* Duration

To check benchmark via OpenVINO:

>NOTE: Before running the tool with a trained model, make sure the model is converted to the Inference Engine format (*.xml +*.bin) using the Model Optimizer tool.

Run the tool with specifying the <INSTALL_DIR>/deployment_tools/demo/car.png file as an input image(can use any image of car), the IR of the ssd_inception_v2_coco model and a device to perform inference on. The following commands demonstrate running the Benchmark Tool in the asynchronous mode on CPU and FPGA devices:
On CPU:
'''
python3 benchmark_app.py -m <IR_dir>/ssd_inception_v2_coco.xml -d CPU -api async -i <INSTALL_DIR>/deployment_tools/demo/car.png --progress true -b 1
'''
>Note: Make sure that benchmark app runs on your device and all necessary files are installed beforehand.

The application outputs number of executed iterations, total duration of execution, latency and throughput. Additionally, if you set the -pc parameter, the application outputs performance counters. If you set '-exec_graph_path', the application reports executable graph information serialized.

Output will be in the form:

>For CPU:
[Step 8/9] Measuring performance (Start inference asyncronously, 60000 ms duration, 4 inference requests in parallel using 4 streams)
Progress: [....................] 100.00% done
[Step 9/9] Dumping statistics report
[ INFO ] Statistics collecting was not requested. No reports are dumped.
Progress: [....................] 100.00% done
Count:      688 iterations
Duration:   60604 ms
Latency:    349.49 ms
Throughput: 11.3 FPS

## Assess Model Use Cases

1. Can be used at polling stations and restrict if more than certain amount of people enters inside station.
2. Can be used in examination hall/malls tracking number of people entering and exiting.
3. To use at marts and also check what part in store gets more visited and also track times when there's a peak.

## Assess Effects on End User Needs

1. Make sure the deployed device is provided with good amount of light as lighting could change the brightness or hue of the images
2. Focal length could make images blurry so it must be focused upto optimal length.
3. Small image size means low resolution and it'll lead to inaccurate results also if the person in image   is enlarged.
4. Very high resolution image is also restricted such as 720p,1080p and 4k(cannot be transmitted).
