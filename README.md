### Fix-Ride - Modal Training

Steps to follow to extablis the API

For Windows:

- python -m venv env
- .\env\Scripts\activate

For Mac/Linux:

- python3 -m venv env
- source env/bin/activate

`pip install fastapi uvicorn tensorflow keras pillow numpy python-multipart`

`uvicorn main:app --reload`
