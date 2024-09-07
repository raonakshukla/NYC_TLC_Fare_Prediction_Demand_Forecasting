# Team 14 Metric Marvins Group Project

### Project Name: Large-Scale Data Analytics on NYC Taxi Limousine Commission(TLC) Data for Fare Prediction and Demand Forecasting
Urban transportation is facing unprecedented challenges: traffic congestion, pollution, and inefficient resource allocation. Taxi and ride-sharing companies generate massive amounts of data on trip patterns, driver behavior, and rider preferences. This study is an effort to provide one platform solution by leveraging these datasets to optimize urban mobility, improve service efficiency, and ultimately create a more sustainable transportation network. The intent behind the project is to empower management in data-driven decision-making. By harnessing the power of data, companies can make smarter decisions, optimize operations, and gain a competitive edge in the marketplace. Weekly demand forecasting is performed using SARIMA models, which scored the lowest on the Akaike Information Criterion (AIC) at 1390, compared to other models. The study aims to predict base fares using various regression models, employing both global and local approaches. Additionally, it will examine the performance of distributed algorithms, with a primary focus on computational power (i.e., the number of available cores) and the size of the data. The Root Mean Square Error(RSME) obtained is the least for 16 partitions (9.34), which is very close to the global model (9.10). However, the time required to train and calculate RSME is 8.21 times less than the global model.
#### File Structure
    ├── Scale-up-experiments *** The codes in this directory took reference from [1] ***
    │   │
    │   ├── local_dt_scaleup.ipynb 
    │   │     (Code for scale-up experiments of a Local Model with Decision Tree Regressor)
    │   │
    │   ├── local_dt_sizeup.ipynb 
    │   │     (Code for size-up experiments of a Local Model with Decision Tree Regressor)
    │   │
    │   ├── local_dt_speedup.ipynb
    │   │     (Code for speed-up experiments of a Local Model with Decision Tree Regressor)
    │   │
    │   ├── local_lr_scaleup.ipynb
    │   │     (Code for scale-up experiments of a Local Model with Linear Regression)
    │   │
    │   ├── local_lr_sizeup.ipynb
    │   │     (Code for size-up experiments of a Local Model with Linear Regression)
    │   │
    │   ├── local_lr_speedup.ipynb
    │   │     (Code for speed-up experiments of a Local Model with Linear Regression)
    │   │
    │   ├── plot_metrics
    │   │     (Code for plotting the results of the experiments)
    │   │     Specify which model you are plotting on top of the code and run all cells.
    │   │
    │   │── Time Series Modelling
    │   │   (Code for Time Series Modelling. export.csv compiled by grouping trip count on week) 
    │   │ 
    │   │── Local Approach and Ensemble Effect
    │   │   (Code for local approach for regression and results to show Ensemble Effect)
    │   │
    │   │── Global Approach
    │   │   (Code for Globa approach for regression and results for three models Linear Regression, Decision Trees and Gradient Boost)
    │   │
    │   │── Heat Maps
    │   │   (Code for Geo Spatial Visaulisation of demand. Like time series files are compiled and exported)
    │   │ 
    │   │ *** below are records from each experiment ***
    │   ├── Global_Ensemble.csv
    │   ├── Global_Ensemble_dt.csv
    │   ├── Global_Ensemble_gbt.csv
    │   ├── sizeup_dt.csv
    │   ├── speedup_dt.csv
    │   ├── scaleup_lr.csv
    │   ├── sizeup_lr.csv
    │   └── speedup_lr.csv
    │
    ├── data_visualization.ipynb
    │   (Code for visualzing the entire HVFHV Data by hourly, daily, monthly, yearly, Ride-share companies)
    │
    ├── eda_heatmap.ipynb
    │   (Code for performing Exploratory Data Analysis and data manipulation for creating a table for a heatmap)
    │
    ├── feature_selection.ipynb
    │   (Code for performing Recursive Feature Elimination)
    │
    └── README.md

### GitLab Repo ###
https://projects.cs.nott.ac.uk/big-data1/team14-metric-marvins-group-project/-/tree/main?ref_type=heads

### How to run ###
All files are in the jupyter notebook format. Therefore, just run the file from top to bottom unless specified in the above file structure diagram.

### Reference ###
[1] I. Triguero and M. Galar, Large-Scale Data Analytics with Python and Spark: A Hands-on Guide to Implementing Machine Learning Solutions. Cambridge: Cambridge University Press, 2024, pp. 229-240. 
