
**All my Notebooks were run on Google Colab with Colab Pro subscription. The CPU type was A-100.**
Even with an A-100 CPU type on Google Colab, the total training time for 5 models took about 27 mins. The EDA also took some time as the pixel intensity analysis is compute intensive.

#### Instructions to Run these Notebooks on Google Colab
1. `capstone_utils.py`  should be copied to the `/content` folder on Google Colab run time.
2. `custom_cinic10_data.zip` should be copied to the `/content` folder on Google Colab run time. The Notebook will unzip this file for performance evaluation.
3. `capstone_utils.py` and `custom_cinic10_data.zip` should be copied to the `/content` folder on Google Colab after a new runtime is started. 


**Note**: Colab deletes all files uploaded to it when a runtime is deleted or the CPU/GPU is disconnected. Re-upload of `capstone_utils.py` and `custom_cinic10_data.zip` (along with unzipping operation) is required in that case.
