FROM python:3.10.20-slim
LABEL org.opencontainers.image.source https://github.com/SENERGY-Platform/leakage-detection
WORKDIR /usr/src/app
# git installs the operator library and records the commit of the build.
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
# The dependencies before the code, so that a change of the code reuses their layer.
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt
COPY . .
RUN git log -1 --pretty=format:"commit=%H%ndate=%cd%n" > git_commit && rm -rf .git && apt-get purge -y git && apt-get autoremove -y
CMD [ "python3", "-u", "main.py"]
