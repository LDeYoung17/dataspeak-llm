FROM alpine

ADD --keep-git-dir=true https://github.com/LDeYoung17/dataspeak-llm/tree/work-branch/buildkit.git#v0.10.1 /buildkit

WORKDIR /src

RUN --mount=target=. \
  make REVISION=$(git rev-parse HEAD) build


EXPOSE 8000

# Start the app using serve command
CMD [ "serve", "-s", "build" ]