FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libopenal1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        fastmcp==2.14.5 \
        gymnasium==1.2.3 \
        imageio-ffmpeg==0.6.0 \
        moviepy==2.1.2 \
        vizdoom==1.3.0

COPY server.py /app/server.py
COPY streaming_video.py /app/streaming_video.py
COPY iwads/ /iwads/
RUN mkdir -p /vizdoom-results

WORKDIR /app
EXPOSE 8000

CMD ["python", "server.py"]
