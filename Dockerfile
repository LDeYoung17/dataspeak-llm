FROM alpine

RUN apk add --no-cache git
RUN git clone --branch v0.10.1 --single-branch https://github.com/LDeYoung17/dataspeak-llm.git /buildkit

WORKDIR /src

EXPOSE 8000

# Start the app using serve command
CMD [ "serve", "-s", "build" ]