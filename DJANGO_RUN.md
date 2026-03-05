# Run Django Frontend

From project root:

```powershell
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python manage.py migrate
.\.venv\Scripts\python manage.py runserver
```

Open:

- http://127.0.0.1:8000/dashboard
- http://127.0.0.1:8000/ranking
- http://127.0.0.1:8000/portfolio
- http://127.0.0.1:8000/risk

Production (example):

```powershell
.\.venv\Scripts\python -m gunicorn stockitup_django.wsgi:application --workers 4 --bind 0.0.0.0:8000
```

