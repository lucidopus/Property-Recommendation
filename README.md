# Property-Recommendation

## Overview

This is a FastAPI-based web service designed to match real estate leads with available rental and resale properties. It performs lead matching based on a set of criteria such as city, unit type, and more. The service uses CSV data files (generated daily) and compares lead parameters with real estate listings to find the most suitable matches. It also provides an API key-based authentication for secure access to its endpoints.

## Features

- **Health Check**: `/health` endpoint to verify if the API is up and running.
- **API Key Authentication**: Secure API endpoints that require an API key for access.
- **Rental Matching**: Matches rental property leads with available listings based on similarity score.
- **Resale Matching**: Matches resale property leads with available listings based on similarity score.
- **Database Logging**: Logs all match attempts and results to a MongoDB database.
- **CSV Handling**: Automatically generates and uses daily CSV files for real estate data.

## Endpoints

### 1. `/health` [GET]

- **Description**: A simple health check endpoint to verify the API status.
- **Response**: 
  ```json
  {
    "status": "ok"
  }

### 2. `/match/rental` [POST]

- **Description**: Matches a rental property lead with the most suitable property based on the given criteria (city, unit type, etc.).
- **Request Body**:
```json
{
"city": "string",
"unit_type": "string",
"other_params": "value"
}
```
- **Response (if match found)**:
```json
{
  "success": true,
  "message": "MATCH FOUND",
  "similarity_score": 45,
  "lead_features": {...},
  "focus_features": {...}
}
```

- **Response (if no match found)**:
```json
{
  "success": false,
  "message": "NO MATCH FOUND",
  "lead_features": {...},
  "focus_features": null
}
```

### 3. `/match/resale` [POST]

- **Description**: Matches a resale property lead with the most suitable property based on the given criteria (city, unit type, etc.).
- **Request Body**:
```json
{
"city": "string",
"unit_type": "string",
"other_params": "value"
}
```

- **Response (if match found)**:
```json
{
  "success": true,
  "message": "MATCH FOUND",
  "similarity_score": 45,
  "lead_features": {
    "city": "city_name",
    "unit_type": "unit_type_name",
    "other_params": "value"
  },
  "focus_features": {
    "city": "city_name",
    "unit_type": "unit_type_name",
    "other_params": "value"
  }
}
```

- **Response (if no match found)**:
```json
{
  "success": false,
  "message": "NO MATCH FOUND",
  "lead_features": {
    "city": "city_name",
    "unit_type": "unit_type_name",
    "other_params": "value"
  },
  "focus_features": null
}
```


