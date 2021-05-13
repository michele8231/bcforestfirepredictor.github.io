from flask import Flask, render_template, request, redirect, url_for, jsonify, flash, session
import datetime
from datetime import timedelta
import pytz
import gmplot
import numpy
import pandas
import requests
import pickle
import sklearn, sklearn.tree, sklearn.ensemble, sklearn.feature_extraction, sklearn.metrics

app = Flask(__name__)
app.secret_key = "haha"
app.permanent_session_lifetime = timedelta(minutes = 5)

apikey = "AIzaSyAw_BHttzAxFdQ4Lg5UT7oYTjJ5LFON2Z8"

#Parts of the ensemble model
rf = pickle.load(open('random_forest.pk1', 'rb'))
ml_reg = pickle.load(open('multilinear_regression.pk1', 'rb'))

#Subtitute for empty data obtained from API
#in the order of: ign_date, latitude, longitude, minimum temperature,maximum temperature, average temperature, dew point, relative humidity, heat index, wind speed, wind gust, wind direction, wind chill, precipitation, visibility, cloud cover, sea level pressure
avg_for_na = [20026124.90970577, 
53.40442305249716, 
-122.14423555004001,
40.58008083140884,
63.09509237875295,
52.084209006928425,
37.187727006444,
62.41874633860559,
85.18183962264142,
12.53463329452857,
27.2919663351186,
192.44023689197786,
27.720631578947398,
0.06612954545454518,
76.20172189733584,
30.337810514153652,
1015.3026817640034]

#Ensemble model
def ensemble_model(input_df):
  dvr = sklearn.feature_extraction.DictVectorizer()
  rf_input = dvr.fit_transform(input_df.T.to_dict().values())
  rf_pred = rf.predict(rf_input) 
  ml_reg_pred = ml_reg.predict(input_df)
  return (rf_pred + ml_reg_pred) / 2.0

# Get tomorrows date of BC in model-friendly format
def get_tommorow_date_BC():
  my_date = datetime.datetime.now(pytz.timezone('US/Pacific')) + datetime.timedelta(days=1)
  proper_month = str(my_date.month)
  proper_day = str(my_date.day)
  if(my_date.month < 10):
    proper_month = '0' + str(my_date.month)
  if(my_date.day < 10):
    proper_day = '0' + str(my_date.day)
  return int(str(my_date.year) + proper_month + proper_day)


def split(word):
  return [char for char in word]

#Use visual crossing weather forecast api to get climate data from user inputs
def get_future_climate_data(latitude, longitude):
  url = "https://visual-crossing-weather.p.rapidapi.com/forecast"
  coords = f'{latitude}, {longitude}'
  querystring = {"location":coords,"aggregateHours":"24","shortColumnNames":"0","unitGroup":"us","contentType":"csv"}

  headers = {
      'x-rapidapi-key': "5416e60fd3msh25313be4cf91e39p1db2f2jsn7acc7269aacf",
      'x-rapidapi-host': "visual-crossing-weather.p.rapidapi.com"
      }

  response = requests.request("GET", url, headers=headers, params=querystring)
  response_list = response.text.split(",")

  return response_list

#Get model-friendly input df with all required data from user inputs (and with the help of visual crossing weather api)
def get_input_df(ign_date, latitude, longitude):
  def to_nan(entry):
    if (entry == '' or entry == 'N/A'): return numpy.nan
    else: return entry

  response_list = get_future_climate_data(latitude, longitude)
  if (len(response_list) < 22):
      response_list = ['N/A'] * 52
  input_data = {'IGN_DATE': ign_date, 'LATITUDE': latitude, 'LONGITUDE': longitude
  , 'minimum temperature': response_list[31], 'maximum temperature': response_list[32]
  , 'average temperature': response_list[33], 'dew point': avg_for_na[6], 'relative humidity': response_list[42]
  , 'heat index': response_list[36], 'wind speed': response_list[34], 'wind gust': response_list[43]
  , 'wind direction': response_list[30], 'wind chill': response_list[44], 'precipitation': response_list[38]
  , 'visibility': avg_for_na[14], 'cloud cover': response_list[35], 'sea level pressure': response_list[39]}

  input_data_frame = pandas.DataFrame(data = input_data, index=[0])
  print(input_data_frame)
  count = 0

  for column in input_data_frame:
    input_data_frame[column] = input_data_frame[column].apply(to_nan) 
    input_data_frame[column].fillna(avg_for_na[count], inplace=True)
    count = count + 1

  return input_data_frame


@app.route('/', methods = ['POST', 'GET'])

def home_page():
  if(request.method == 'POST'):
      longitude_cord = request.form['longitude_textbar']
      latitude_cord = request.form['latitude_textbar']
      if(-130.0 <= float(longitude_cord) and float(longitude_cord) <= -115.0 and float(latitude_cord) >= 48.0 and float(latitude_cord) <= 60.0):
        return redirect(url_for("map_page", longitude = longitude_cord, latitude = latitude_cord))
      else:
        flash("Error: Coordinates do not point to a location in/near to BC, try again", "info")
        return redirect(url_for('home_page'))
  else:
      return render_template('webpage.html')
    

@app.route('/result')

def map_page():
    tommorow_date_BC = get_tommorow_date_BC()
    longitude = float(request.args.get('longitude', None))
    latitude = float(request.args.get('latitude', None))
    longitudes_within_range = [longitude, longitude+0.1, longitude-0.1]
    latitudes_within_range = [latitude, latitude+0.1, latitude-0.1]
    threshold_fire = numpy.log(3)
    lower_than_threshold = []
    higher_than_threshold = []
    size_of_fire = []

    for longitude_in_list in longitudes_within_range:
        for latitude_in_list in latitudes_within_range:
            input_for_model = get_input_df(tommorow_date_BC, latitude_in_list, longitude_in_list)
            prediction = ensemble_model(input_for_model)
            print(prediction)
            if(prediction > threshold_fire):
                higher_than_threshold.append((latitude_in_list, longitude_in_list))
                size_of_fire.append(prediction)
            else :
                lower_than_threshold.append((latitude_in_list, longitude_in_list))

    gmap = gmplot.GoogleMapPlotter(latitude, longitude, 10, apikey=apikey)

    if(len(lower_than_threshold) > 0):
      no_fire_lats, no_fire_lngs = zip(*lower_than_threshold)
      gmap.scatter(no_fire_lats, no_fire_lngs, color='#808080', size=1000, marker=False)
    
    if(len(higher_than_threshold) > 0):
      for i in range(len(higher_than_threshold)):
        yes_fire_lats, yes_fire_lngs = zip(*[higher_than_threshold[i]])
        gmap.scatter(yes_fire_lats, yes_fire_lngs, color='#FF0000', size= 1000, marker=False)
        gmap.marker(higher_than_threshold[i][0], higher_than_threshold[i][1],title="Approx-size: " + str(numpy.exp(size_of_fire[i]) - 1) + " Hectares")
    
    area = zip(*[(latitude-0.1, longitude-0.1), (latitude+0.1, longitude-0.1), (latitude+0.1, longitude+0.1), (latitude-0.1, longitude+0.1), 
    (latitude-0.1, longitude-0.1), (latitude, longitude-0.1), (latitude, longitude+0.1), (latitude+0.1, longitude+0.1), (latitude+0.1, longitude), (latitude-0.1, longitude)])
    gmap.plot(*area, color='cornflowerblue', edge_width=1)
    
    gmap.draw('templates/map.html')
    return render_template('map.html')
    
    

if __name__ == "__main__":
    app.run(debug=True)