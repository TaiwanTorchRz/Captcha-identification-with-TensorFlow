�	��y��A@��y��A@!��y��A@	sp&�9�?sp&�9�?!sp&�9�?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$��y��A@Tt$����?A��S㥳A@Y�c�ZB�?*	33333�Z@2f
/Iterator::Model::Prefetch::MapAndBatch::Shuffle���Q��?!��s�M�K@)���Q��?1��s�M�K@:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatchA��ǘ��?!��K3��4@)A��ǘ��?1��K3��4@:Preprocessing2F
Iterator::Model-C��6�?!|��h�7@)�5�;Nё?1�"��$0@:Preprocessing2P
Iterator::Model::Prefetch	�^)ˀ?!>G�D=m@)	�^)ˀ?1>G�D=m@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	Tt$����?Tt$����?!Tt$����?      ��!       "      ��!       *      ��!       2	��S㥳A@��S㥳A@!��S㥳A@:      ��!       B      ��!       J	�c�ZB�?�c�ZB�?!�c�ZB�?R      ��!       Z	�c�ZB�?�c�ZB�?!�c�ZB�?JCPU_ONLY2black"�
device�Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"CPU: 