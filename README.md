# Analysis of Methods for Optical-Band Light Curve
## Overview
**Abstract**: With the continuous development of research on optical-band light curves in astronomy, an increasing number of time series analysis techniques have been employed to reveal temporal patterns and underlying physical mechanisms of celestial objects. However, due to the different data characteristics and algorithmic ideas, it is rather difficult to select an appropriate method in practical tasks. So this paper provides an analysis of time domain techniques for optical-band light curve from two parts. Firstly, we categorize existing light curve analysis methods into six major classes and conduct an in-depth review of each category, including their fundamental concepts, advantages, caveats, and representative applications. Second, we construct seven datasets based on stellar types using light curve data from the OGLE-IV survey to uniformly evaluate the prediction performance of seven classical algorithms. Additionally, we assess the classification performance of four representative methods using the PLAsTiCC dataset. 
## Datasets Download
Please download the data from the following link.</br>
OGLE:  [OGLE](https://pan.quark.cn/s/1e55ebc5aaf1)</br>
PLAsTiCC:  [PLAsTiCC_Website](https://zenodo.org/records/2539456) or [PLAsTiCC_DirectDownload](https://zenodo.org/api/records/2539456/files-archive)</br>
## Code Guidance
Overall project structure:
```text
----   
    |----README.md
    |----ARIMA_Predictor.py # model ARIMI to predict
    |----GRU_Predictor.py # model GRU to predict
    |----CNN_Predictor.py # model CNN to predict
    |----LSTM_Predictor.py # model LSTM to predict
    |----RNN _Predictor.py # model RNN to predict
    |----SVR_Predictor.py # model SVR to predict
    |----CNN_Classifier.py # model CNN to classify
    |----GRU_Classifier.py # model SVR to classify
    |----LSTM_Classifier.py # model LSTM to classify
    |----SVM_Classifier.py # model SVM to classify
---- 
```
