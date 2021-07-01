# Predict Level of Drought using Weather Data


## Context
The dataset is derived from the <a href="https://droughtmonitor.unl.edu">US drought monitor</a> The aim of the project is to help investigate if drought can be predicted based on the dataset, potentially leading to generalization of US prediction to other countries. 

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

##


