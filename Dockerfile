FROM alpine

# Clone the repository to your local machine
RUN apk add --no-cache git
RUN git clone --single-branch --branch work-branch https://github.com/LDeYoung17/dataspeak-llm.git /buildkit

WORKDIR /src