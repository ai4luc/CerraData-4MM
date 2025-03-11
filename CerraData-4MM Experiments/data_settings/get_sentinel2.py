import os
from datetime import date, timedelta
import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape

# copernicus User email
copernicus_user = "#"
# copernicus User Password
copernicus_password = "#"
# WKT Representation of BBOX of AOI
ft = "POLYGON ((-49.52605078855572 -5.0331624789304215, -49.52605078855572 -9.963749831522662, -46.040522177447826 -9.963749831522662, -46.040522177447826 -5.0331624789304215, -49.52605078855572 -5.0331624789304215))"
# Sentinel satellite that you are interested in
data_collection = "SENTINEL-2"

today = date.today()
today_string = today.strftime("%Y-%m-%d")
yesterday = today - timedelta(days=30)
yesterday_string = yesterday.strftime("%Y-%m-%d")


def get_keycloak(username: str, password: str) -> str:
    data = {
        "client_id": "cdse-public",
        "username": username,
        "password": password,
        "grant_type": "password",
    }
    try:
        r = requests.post(
            "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
            data=data,
        )
        r.raise_for_status()
    except Exception as e:
        raise Exception(
            f"Keycloak token creation failed. Reponse from the server was: {r.json()}"
        )
    return r.json()["access_token"]


json_ = requests.get(
    f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Collection/Name eq '{data_collection}' and OData.CSC.Intersects(area=geography'SRID=4326;{ft}') and ContentDate/Start gt {yesterday_string}T00:00:00.000Z and ContentDate/Start lt {today_string}T00:00:00.000Z&$count=True&$top=1000"
).json()
print(json_)
p = pd.DataFrame.from_dict(json_["value"])  # Fetch available dataset
if p.shape[0] > 0:
    p["geometry"] = p["GeoFootprint"].apply(shape)
    productDF = gpd.GeoDataFrame(p).set_geometry("geometry")  # Convert PD to GPD
    productDF = productDF[~productDF["Name"].str.contains("L1C")]  # Remove L1C dataset
    print(f" total L2A tiles found {len(productDF)}")
    productDF["identifier"] = productDF["Name"].str.split(".").str[0]
    allfeat = len(productDF)

    if allfeat == 0:
        print("No tiles found for today")
    else:
        ## download all tiles from server
        for index, feat in enumerate(productDF.iterfeatures()):
            try:
                session = requests.Session()
                keycloak_token = get_keycloak(copernicus_user, copernicus_password)
                session.headers.update({"Authorization": f"Bearer {keycloak_token}"})
                url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({feat['properties']['Id']})/$value"
                response = session.get(url, allow_redirects=False)
                while response.status_code in (301, 302, 303, 307):
                    url = response.headers["Location"]
                    response = session.get(url, allow_redirects=False)
                print(feat["properties"]["Id"])
                file = session.get(url, verify=False, allow_redirects=True)

                with open(
                        f"{feat['properties']['identifier']}.zip",  # location to save zip from copernicus
                        "wb",
                ) as p:
                    print(feat["properties"]["Name"])
                    p.write(file.content)
            except:
                print("problem with server")
else:
    print('no data found')
