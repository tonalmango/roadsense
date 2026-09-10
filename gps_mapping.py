"""Road Sense - GPS Mapping Module"""

import json
from pathlib import Path


# --------------------------------------------------
# DEMO GPS DATA
# --------------------------------------------------

GPS_DATA = {
    "road1.jpeg.jpeg": {
        "latitude": 20.2961,
        "longitude": 85.8245
    },

    "road2.jpeg.jpeg": {
        "latitude": 20.2965,
        "longitude": 85.8250
    },

    "road3.jpeg.jpeg": {
        "latitude": 20.2970,
        "longitude": 85.8255
    },

    "road4.jpeg.jpeg": {
        "latitude": 20.2975,
        "longitude": 85.8260
    }
}


# --------------------------------------------------
# GET GPS LOCATION
# --------------------------------------------------

def get_gps(image_name):

    if image_name in GPS_DATA:
        return GPS_DATA[image_name]

    return {
        "latitude": None,
        "longitude": None
    }


# --------------------------------------------------
# SAVE GPS DATA
# --------------------------------------------------

def save_gps_data(output_file="gps/gps_data.json"):

    output_path = Path(output_file)

    # Create GPS folder
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_path, "w") as file:

        json.dump(
            GPS_DATA,
            file,
            indent=4
        )

    print("GPS data saved to:", output_path)


# --------------------------------------------------
# TEST MODULE
# --------------------------------------------------

if __name__ == "__main__":

    print("Road Sense GPS Mapping Module")
    print("-----------------------------")

    for image_name, location in GPS_DATA.items():

        print(
            f"{image_name} | "
            f"Latitude: {location['latitude']} | "
            f"Longitude: {location['longitude']}"
        )

    save_gps_data()

    print("-----------------------------")
    print("GPS mapping completed!")
