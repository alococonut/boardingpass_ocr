export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
uvicorn main:app --host='0.0.0.0' --port=6000 --workers=4 --access-log --backlog=2048 --limit-concurrency=2048 --limit-max-requests=2048