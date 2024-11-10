import os
from datetime import date, timedelta

import pandas as pd
from fastapi import status, FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse

from database import db
from config import (
    PROPMATCH_TOKEN,
    ACCEPTABLE_CITIES_RESALE,
    ACCEPTABLE_CITIES_RENTAL,
    ACCEPTABLE_UNIT_TYPES,
)

from models import LeadRequest, SuccessResponse, FailResponse
from utils import (
    prepare_csv,
    select_subset,
    inference_pipeline_resale,
    inference_pipeline_rental,
)


app = FastAPI(title="PropMatch API", version="1.1.0")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok"}


async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header:
        if api_key_header == PROPMATCH_TOKEN:
            return PROPMATCH_TOKEN
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key"
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please enter an API key",
        )


@app.post("/match/rental", tags=["rental"])
async def solve(request: LeadRequest, token: str = Depends(get_api_key)):
    try:
        lead_params = request.model_dump()
        assert lead_params["city"] in ACCEPTABLE_CITIES_RENTAL, "INVALID CITY"
        assert lead_params["unit_type"] in ACCEPTABLE_UNIT_TYPES, "INVALID UNIT TYPE"
        filename = (
            "_".join(["Rental", date.today().strftime(format="%d-%m-%Y")]) + ".csv"
        )

        # below snippet is to fetch and prepare csv once everyday and then reuse it
        if os.path.exists(filename):
            rental_data = pd.read_csv(filename)
        else:
            filename = prepare_csv("Rental")
            rental_data = pd.read_csv(filename)
            previous_filename = (
                "_".join(
                    [
                        "Rental",
                        (date.today() - timedelta(days=1)).strftime(format="%d-%m-%Y"),
                    ]
                )
                + ".csv"
            )
            if os.path.exists(previous_filename):
                os.remove(previous_filename)

        rental_subset = select_subset(
            rental_data, lead_params["city"], lead_params["unit_type"]
        )
        similar_projects = inference_pipeline_rental(
            lead_params=lead_params, focus_data=rental_subset
        )
        if similar_projects["similarity_score"].iloc[1] >= 35:
            status = {"success": True, "message": "MATCH FOUND"}
            most_similar = similar_projects.iloc[1].to_dict()
            response = {**status, **most_similar}
            db["records"].insert_one(
                {
                    **status,
                    **{"lead_features": lead_params, "focus_features": most_similar},
                }
            )

            return SuccessResponse(**response)
        else:
            response = {"success": False, "message": "NO MATCH FOUND"}

            db["records"].insert_one(
                {**response, **{"lead_features": lead_params, "focus_features": None}}
            )
            return FailResponse(**response)
    except Exception as e:
        response = {
            "success": False,
            "message": f"ERROR OCCURED: {str(e)}",
        }
        db["records"].insert_one(
            {**response, **{"lead_features": lead_params, "focus_features": None}}
        )
        return JSONResponse(content=response, status_code=500)


@app.post("/match/resale", tags=["resale"])
async def solve(request: LeadRequest, token: str = Depends(get_api_key)):
    try:
        lead_params = request.model_dump()
        assert lead_params["city"] in ACCEPTABLE_CITIES_RESALE, "INVALID CITY"
        assert lead_params["unit_type"] in ACCEPTABLE_UNIT_TYPES, "INVALID UNIT TYPE"
        filename = (
            "_".join(["Resale", date.today().strftime(format="%d-%m-%Y")]) + ".csv"
        )

        # below snippet is to fetch and prepare csv once everyday and then reuse it
        if os.path.exists(filename):
            resale_data = pd.read_csv(filename)
        else:
            filename = prepare_csv("Resale")
            resale_data = pd.read_csv(filename)
            previous_filename = (
                "_".join(
                    [
                        "Resale",
                        (date.today() - timedelta(days=1)).strftime(format="%d-%m-%Y"),
                    ]
                )
                + ".csv"
            )
            if os.path.exists(previous_filename):
                os.remove(previous_filename)

        resale_subset = select_subset(
            resale_data, lead_params["city"], lead_params["unit_type"]
        )
        similar_projects = inference_pipeline_resale(
            lead_params=lead_params, focus_data=resale_subset
        )
        if similar_projects["similarity_score"].iloc[1] >= 40:
            status = {"success": True, "message": "MATCH FOUND"}
            most_similar = similar_projects.iloc[1].to_dict()
            response = {**status, **most_similar}
            db["records"].insert_one(
                {
                    **status,
                    **{"lead_features": lead_params, "focus_features": most_similar},
                }
            )

            return SuccessResponse(**response)
        else:
            response = {"success": False, "message": "NO MATCH FOUND"}

            db["records"].insert_one(
                {**response, **{"lead_features": lead_params, "focus_features": None}}
            )
            return FailResponse(**response)
    except Exception as e:
        response = {
            "success": False,
            "message": f"ERROR OCCURED: {str(e)}",
        }
        db["records"].insert_one(
            {**response, **{"lead_features": lead_params, "focus_features": None}}
        )
        return JSONResponse(content=response, status_code=500)
