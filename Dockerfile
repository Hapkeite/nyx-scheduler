FROM pytorch/pytorch:latest
RUN apt-get update && apt-get install -y g++ nvidia-cuda-dev
WORKDIR /app
COPY . /app
RUN cd src/interceptor && g++ -O3 -fPIC -shared minimal_intercept.cpp -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -ldl -o liborion_capture.so
