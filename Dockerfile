from python:3

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

COPY Cat03.jpg /tmp/image

CMD ["python", "-u", "rhel_demo_worker.py"]
