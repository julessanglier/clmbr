from typing import Union, Optional

from fastapi import FastAPI, HTTPException

from v3 import find_the_roads

app = FastAPI()

@app.get("/find", tags=["data"])
def find(city: str, k: Optional[int] = None, r: Optional[float] = None,
         mas: Optional[float] = None, ml: Optional[float] = None, no: Optional[bool] = None):
    """Search for steep roads.

    Query params:
    - city (str): city name (required)
    - k (int): optional limit of returned results
    - r (float): radius in meters
    - mas (float): min average slope percent
    - ml (float): min segment length in meters
    - no (bool): named-only, if true, exclude unnamed roads
    """
    try:
        results = find_the_roads(city=city, k=k, r=r, mas=mas, ml=ml, no=no)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))