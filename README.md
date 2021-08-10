# Predict Level of Drought using Weather Data


## Context
This proejct is based on Kaggle project - Predict Droughts using Weather & Soil Data. The dataset is derived from the <a href="https://droughtmonitor.unl.edu">US drought monitor</a> The aim of the project is to help investigate if drought can be predicted based on the dataset, potentially leading to generalization of US prediction to other countries. 

## Dataset
This is a classification dataset over 6 levels of drought, from 0 (no drought) to 5 (exceptional drought). Each entry is a drought level at specific time point in a particular region in US, accompanied by the last 90 days of 18 meteorological indicators.<br/>

### Drought Level
![inbox-2055480-f5ad8544ab11d043972fb9209a874dd3-levels](https://user-images.githubusercontent.com/43593664/124084318-7300ca00-da81-11eb-9c99-c59875ad01fa.PNG)


### Meteorological Indicators
<table>
<thead>
<tr>
<th>Indicator</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>WS10M_MIN</td>
<td>Minimum Wind Speed at 10 Meters (m/s)</td>
</tr>
<tr>
<td>QV2M</td>
<td>Specific Humidity at 2 Meters (g/kg)</td>
</tr>
<tr>
<td>T2M_RANGE</td>
<td>Temperature Range at 2 Meters (C)</td>
</tr>
<tr>
<td>WS10M</td>
<td>Wind Speed at 10 Meters (m/s)</td>
</tr>
<tr>
<td>T2M</td>
<td>Temperature at 2 Meters (C)</td>
</tr>
<tr>
<td>WS50M_MIN</td>
<td>Minimum Wind Speed at 50 Meters (m/s)</td>
</tr>
<tr>
<td>T2M_MAX</td>
<td>Maximum Temperature at 2 Meters (C)</td>
</tr>
<tr>
<td>WS50M</td>
<td>Wind Speed at 50 Meters (m/s)</td>
</tr>
<tr>
<td>TS</td>
<td>Earth Skin Temperature (C)</td>
</tr>
<tr>
<td>WS50M_RANGE</td>
<td>Wind Speed Range at 50 Meters (m/s)</td>
</tr>
<tr>
<td>WS50M_MAX</td>
<td>Maximum Wind Speed at 50 Meters (m/s)</td>
</tr>
<tr>
<td>WS10M_MAX</td>
<td>Maximum Wind Speed at 10 Meters (m/s)</td>
</tr>
<tr>
<td>WS10M_RANGE</td>
<td>Wind Speed Range at 10 Meters (m/s)</td>
</tr>
<tr>
<td>PS</td>
<td>Surface Pressure (kPa)</td>
</tr>
<tr>
<td>T2MDEW</td>
<td>Dew/Frost Point at 2 Meters (C)</td>
</tr>
<tr>
<td>T2M_MIN</td>
<td>Minimum Temperature at 2 Meters (C)</td>
</tr>
<tr>
<td>T2MWET</td>
<td>Wet Bulb Temperature at 2 Meters (C)</td>
</tr>
<tr>
<td>PRECTOT</td>
<td>Precipitation (mm day-1)</td>
</tr>
</tbody>
</table>

## Data Preprocessing

### Take average value per week

As the score of drought level is recored on Tuesday only, we took the average of other indicators in the past week as our new dataframe for prediction of drought.

<br/><img width="441" alt="image" src="https://user-images.githubusercontent.com/43593664/128814783-d4a1403f-f081-4758-b4a1-7e9805d21794.png">

### Round the level of drought
Some of the level were found to be between 0 to 1. To classify the levels into 6 bins precisely (0-5), these levels were rounded.
<br/>
Before:
<br/><img width="350" alt="Screenshot 2021-08-10 at 2 23 11 PM" src="https://user-images.githubusercontent.com/43593664/128818359-eb939e8a-7777-491e-9399-c3ec72b3d994.png">
<br/>After:
<br/><img width="349" alt="image" src="https://user-images.githubusercontent.com/43593664/128816641-797127c2-6d28-4cd6-9201-e3937aed61eb.png">

## Models
We took the 18 meteorological indicators as X and the level of drought as y. 

## Model Evaluation

### SGDclassifier
<br/><img width="470" alt="image" src="https://user-images.githubusercontent.com/43593664/128818951-c95e16a5-d79c-4024-a017-b8db51270a64.png">
<br/><img width="470" alt="image" src="https://user-images.githubusercontent.com/43593664/128818993-0b017444-efb4-41a6-a1a3-16f49fcce242.png">

### Decision Tree
<br/><img width="470" alt="image" src="https://user-images.githubusercontent.com/43593664/128819019-ae11dbe3-eadb-4da3-8d79-2633506a8fcf.png">
<br/><img width="470" alt="image" src="https://user-images.githubusercontent.com/43593664/128819055-60c36c46-16c7-4d46-b65b-6b1f31a0c980.png">

### Random Forest
<br/><img width="470" alt="image" src="https://user-images.githubusercontent.com/43593664/128819100-153204ec-a1ef-4547-8a89-622c009c52c2.png">

### XGBoost
<br/><img width="470" alt="image" src="https://user-images.githubusercontent.com/43593664/128819129-42c14f2a-dc84-4f14-88f4-fa91bf420947.png">
<br/><img width="470" alt="image" src="https://user-images.githubusercontent.com/43593664/128819159-58af2f84-c96a-4cdf-8bd8-f9e63aeccae5.png">
