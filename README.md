# Visual Relationship Detection
Two paper results for Visual relationship detection. 

This work is done by Caffe 1 with Python custom layer.

The first paper : Visual Relationship Detection with Language prior and Softmax accepted in IPAS 2018
The Second paper : Improving Visual Relationship Detection uisng Linguistic and Spatail Cues submmitted in ETRI Journal

The urls for two papers will be open soon.

<img src="https://github.com/Jungjaewon/Visual-Relationship-Detection/blob/master/overview.png" width="400">

**Predicate Predication**

| Model         | R@50,k=1      | R@50,k=70     | R@50,k=1,z    | R@50,k=70,z   |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| VRD           | 47.87         | -             | 8.45          | -             |
| U+W+SF+L:S    | 47.50         | 74.98         | 16.98         | 54.20         |
| U+W+SF+L:T    | 54.13         | 82.54         | 8.80          | 32.81         |
| U+W+SF+L:T+S  | 55.16         | 85.64         | -             | -             |
| R-S           | 51.50         | -             | 14.60         | -             |
| LS+SVW        | 55.16         | 88.88         | 21.38         | 64.50         |

**Spatial Vector Proof**

| Model         | R@50,k=1      | R@50,k=70     | R@50,k=1,z    | R@50,k=70,z   |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| (SF)VW        | 45.58         | 77.10         | 13.60         | 53.63         |
| L(SF)+(SF)VW  | 50.53         | 81.99         | 15.99         | 56.33         |
| SVW           | 48.75         | 78.04         | 16.85         | 55.77         |
| LS+SVW        | 55.16         | 88.88         | 21.38         | 64.50         |

**Spatial Module Proof**

| Model         | R@50,k=1      | R@50,k=70     | R@50,k=1,z    | R@50,k=70,z   |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| LS+SVW        | 55.16         | 88.88         | 21.38         | 64.50         |
| *             | 53.99         | 87.26         | 22.75         | 66.38         |
| T 0.1 *       | 55.16         | 88.84         | 21.38         | 64.49         |
| T 0.2 *       | 55.25         | 88.09         | 22.24         | 62.87         |
| T 0.3 *       | 55.19         | 84.69         | 22.49         | 60.51         |
| T 0.4 *       | 54.77         | 82.42         | 23.09         | 58.51         |

