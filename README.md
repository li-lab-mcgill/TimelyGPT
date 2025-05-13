# TimelyGPT

TimelyGPT is a extrapolatable time-serie generative pre-training Transformer designed for representation learning and health forecasting. 
This model is designed to model both continuous time series (e.g., biosignals) and irregularly-sampled time series (e.g., longitudinal electronic health records).


# Overview

This figure indicates an overview of TimelyGPT architecture.

***Panel a***. TimelyGPT Architecture. It shows that an example of continuous time-series data is first tokenized by a two-layer 1D convolutional subsampler and then projected into a $d$-dimensional space. A stack of $L$ generative decoder layers then processes it, and an output projection finally produces the next-token predictions.

***Panel b***. Each decoder layer is coupled with xPos embedding that encodes trend and periodic patterns into time-series representations, facilitating forecasting with extrapolation ability.

***Panel c***. Chunk-wise Retention consists of parallel intra-chunk Retention and recurrent inter-chunk Retention, effectively handling long sequences in continuously monitored biosignals

***Panel d***. Temporal Convolution captures nuanced local interactions from time-series representations.

<img src=https://github.com/li-lab-mcgill/TimelyGPT/blob/master/figures/architecture.png width="800">


# Code Organization

***Continuous Time Series (CTS)***. For the continous time series (e.g., biosignals), you can use the code under the folder ```TimelyGPT_CTS```.

Particularly, you need to place the data under the folder ```TimelyGPT_CTS/data``` and modify the corresponding argument code
```parser.add_argument('--data_folder', type=str, default='sleepEDF', help='data file')```.
    

***Irregularly-sampled Time Series (ISTS)***. For the irregularly-sampled time series  (e.g., longitudinal electronic health records), you can use the code under the folder ```TimelyGPT_ISTS```.

Particularly, you need to place the data under the folder ```TimelyGPT_ISTS/data``` and modify the corresponding argument code
```parser.add_argument('--data_path', type=str, default='processed_pophr_data.csv', help='data file')```.



# Relevant Publications

This published code is referenced from: 

***Ziyang Song***, Qincheng Lu, Hao Xu, Mike He Zhu, David Buckeridge, and Yue Li. (2024).
TimelyGPT: Extrapolatable Transformer Pre-training for Long-term Time-Series Forecasting in Healthcare.
The 15th ACM Conference on Bioinformatics, Computational Biology, and Health Informatics (ACM BCB).
