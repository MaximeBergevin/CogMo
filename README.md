# CogMo

CogMo is a specialized tool for batch processing and visualizing force data exported from any data acquisition software. The CogMo Tool provides a dashboard for visual inspection of every trial in a novel scientific paradigm that manipulates both the motor and cognitive demands of a psychomotor experimental task.

Several key metrics are measured from force traces (e.g., peak force, rate of force development, response latency) and electromyography (EMG) traces (root mean square, pre-motor response latency). It is designed to save time during data processing after data acquisition. Although developed with data from the upper limbs (specifically, handgrip and wrist flexion/extension), the app can be used with any muscle group as long as the paradigm is correctly applied.

## Task description
The CogMo paradigm is designed to independently manipulate motor and cognitive workloads. Below is an image example of the task. Note that while the motor demand **must** be manipulated via contraction intensity, cognitive demand can be manipulated in a number of ways. As such, the arrow task is an example. Similarly, the paradigm may be used with various muscle group and is not restricted to a prehension task.

<img src="fig/CogMo-fig_task.png" width="600">

### **Why use the CogMo toolkit?**
Obviously, this paradigm provides behavioural data that can be collected with any experimental software (e.g., SuperLab, E-Prime, PsychoPy). This app allows you to access more behavioural data by directly processing the force trace. For instance, a typical experimental software would only provide accuracy and reaction time. With the force traces, it is possible to dissociate reaction time (when the force begins to rise) from the response time (when the response is validated). It is also possible to derive other interesting measures, like whether participants over- or undershoot the threshold in specific conditions. In addition, you can choose to upload EMG traces alongside the force traces for additional processing and gain more insights into the neurophysiological systems during the paradigm. Currently, only EMG RMS for each contraction burst, as well as the pre-motor reaction time, are available. See figure below.

<img src="fig/CogMo-fig_analyses.png" width="900">

