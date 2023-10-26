FROM alpine

RUN apk add --no-cache git
RUN git clone --branch main --single-branch https://github.com/LDeYoung17/dataspeak-llm.git /buildkit

EXPOSE 8080

# Start the app using chainlit run command
CMD [ "chainlit run" "app.py"]