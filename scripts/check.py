import requests

for url in ["http://localhost:8000/health", "http://localhost:8000/stats"]:
    r = requests.get(url, timeout=10)
    print(url, r.status_code, r.text)
