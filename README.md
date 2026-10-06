# Thai category prediction API

## Deploy to Render

1. Push this repository to GitHub, then create a Render Web Service from it.
2. Select `Python 3` as the runtime and `Singapore` as the region. The current NumPy inference build runs on the Free instance (512 MB RAM), but free services spin down after inactivity and the first request can be delayed by 50 seconds or more. Upgrade to at least `2 GB` RAM if the service runs out of memory.
3. Set the build command to `pip install -r requirements.txt`.
4. Set the start command to `uvicorn predict_api:app --host 0.0.0.0 --port $PORT`.
5. Add `PYTHON_VERSION` with value `3.12.10` and `API_KEY` with a long random secret in the service's environment settings. Do not commit the key.
6. Set the health check path to `/health`, deploy, and verify `https://<service-url>/health` returns `{"status":"ok"}`.

## Call from n8n

Create an HTTP Request node with:

- Method: `POST`
- URL: `https://<service-url>/predict`
- Header: `X-API-Key`, stored as an n8n credential
- Body content type: JSON
- Body field: `text` with the expression `{{$json.Text}}`

The response contains `category_id`, `category`, and `confidence`.