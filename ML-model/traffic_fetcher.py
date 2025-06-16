import requests
import csv 
import time
from datetime import datetime
import os



#default testing routes
routes = [
    ("Connaught Place, Delhi", "India Gate, Delhi"),
    ("Rajiv Chowk Metro Station, Delhi", "Hauz Khas, Delhi"),
    ("Noida Sector 18", "South Extension Market, Delhi"),
]

#function to fetch traffic data 

def fetch_traffic_data():
    data = []
    for origin , destination in routes :
        url = (
              f"https://maps.googleapis.com/maps/api/directions/json?"
              f"origin={origin}&destination={destination}&departure_time=now&key={API_KEY}"
        )

        response = requests.get(url)
        result = response.json()

        if result["status"] == "OK":
            leg = result["routes"][0]["legs"][0]
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            distance = leg["distance"]["text"]
            duration  = leg["duration"]["text"]
            trafffic_duration = leg["duration_in_traffic"]["text"]

            #appending the fecthed data from reuslt as using google api
            data.append([timestamp, origin, destination, distance, duration, trafffic_duration])
        else:
            print(f"Error fetching data for {origin} to {destination}:  {result['status']}")
    return data

#function to write data to csv file and save it in ML folder
def save_to_csv(data, filename = "../data/raw/india_traffic.csv"):
     write_header = not os.path.exists(filename)
     with open(filename, mode ="a", newline = "") as file:
               writer = csv.writer(file)
               if write_header:
                 writer.writerow(["timestamp", "origin", "destination", "distance", "duration", "duration_in_traffic"])
               for row in data:
                 writer.writerow(row)
        
#main guard method to run file script directly or imported
if __name__ == "__main__":
    filename = "../data/raw/india_traffic.csv"
    for i in range(6):  # Loop 6 times, 1 per 10 min (1 hour)
        traffic_data = fetch_traffic_data()
        save_to_csv(traffic_data, filename)
        print(f"[{i+1}/6] ✅ Collected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        time.sleep(600)
        #this means our loop  runs 6 time  means in 1 hour it runs 6 times each after every 10 mins