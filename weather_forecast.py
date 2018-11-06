import datetime
import json

import pyowm

# -------------------------------------------------
# Connect to Open Weather Map free account
# to get forecast for Montreal.
#
# Parse and save to a json file to be read
# later to supply weather data to the model input
#
# https:///openweathermap.org
#
# PyOWN module instalation:
#   pip install pyowm
# -------------------------------------------------

result = {
    "date": datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
    "day_precipitation_mm": 0,
    "hourly_temp": {}
}

# get a handler using my api key
owm = pyowm.OWM('1dc779244c89ccffd3d0d0cda8f3a664')

# get forecast of the next 5 days
fcst = owm.three_hours_forecast('Montreal,CA')
forecast = fcst.get_forecast()

# parse the next 24 hours (8 x 3 hours)
for i in range(8):
    weather = forecast.get(i)

    # get time and temperature for this weather forecast object
    temp = weather.get_temperature('celsius')['temp']
    date = weather.get_reference_time(timeformat='date') - datetime.timedelta(hours=4)

    # compute total day precipitation
    precipitation_mm = 0
    status = weather.get_status()
    if status == 'Rain':
        precipitation_mm = weather.get_rain()['3h']
    if status == 'Snow':
        precipitation_mm = weather.get_snow()['3h']
    result["day_precipitation_mm"] += precipitation_mm

    # fill hourly temperature array
    h = date.hour
    result["hourly_temp"][h] = temp
    h += 1
    result["hourly_temp"][h if h <= 24 else (h - 24)] = temp
    h += 1
    result["hourly_temp"][h if h <= 24 else (h - 24)] = temp

    print (date.hour, temp, status, precipitation_mm)

print('-------------------------------')
print(result)

with open('weather_forecast.json', 'w') as outfile:
    json.dump(result, outfile)
