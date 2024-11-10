import os

PROPMATCH_TOKEN = os.environ.get("PROPMATCH_TOKEN")

APP_NAME = os.environ.get("APP_NAME")

GOOGLE_JSON_PATH = os.getenv("GOOGLE_JSON_PATH")
BUCKET_NAME = os.getenv("BUCKET_NAME")

FOCUS_RESALE_PATH = os.getenv("FOCUS_RESALE_PATH")
FOCUS_RENTAL_PATH = os.getenv("FOCUS_RENTAL_PATH")

ACCEPTABLE_UNIT_TYPES = ["Residential"]
ACCEPTABLE_CITIES_RESALE = [
    "Mumbai",
    "Navi Mumbai",
    "Pune",
    "Thane",
    "Gurgaon",
    "Delhi",
    "Noida",
    "Greater Noida",
    "Hyderabad",
    "Bangalore",
    "Chennai",
    "Lucknow",
    "Vizag",
    "Vijaywada",
]
ACCEPTABLE_CITIES_RENTAL = [
    "Mumbai",
    "Thane",
    "Bangalore",
    "Pune",
    "Gurgaon",
    "Hyderabad",
    "Delhi",
    "Noida",
    "Dehradun",
    "Greater Noida",
    "Navi Mumbai",
]
