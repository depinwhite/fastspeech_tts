FROM gcr.io/deeplearning-platform-release/pytorch-gpu
WORKDIR /app/fastspeech

# Copy the contents of repository
COPY . .

USER root

RUN chmod 777 /app/fastspeech/install.sh

RUN  apt-get update \
  && apt-get install -y wget \
  && rm -rf /var/lib/apt/lists/*

RUN /app/fastspeech/install.sh

# Expose port
EXPOSE 3000