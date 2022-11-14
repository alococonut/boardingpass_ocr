export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
#By default, a file named gunicorn.conf.py will be read from the same directory where gunicorn is being run.
#gunicorn main:app -c gunicorn_conf.py
gunicorn main:app --bind 0.0.0.0:6000  --workers 4 --threads 4 --worker-class uvicorn.workers.UvicornWorker