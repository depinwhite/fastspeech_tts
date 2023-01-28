# fastspeech_tts
Fastspeech implementation.

To use this repository run:  
```git clone https://github.com/depinwhite/fastspeech_tts.git```

In order to install all the necessary libraries, you need to execute the following commands:  
```docker build -t fastspeech:latest   --label “autoheal”=“true” -f Dockerfile  .```
```docker run -ti --rm -v LOCAL_MODEL_DIRECTORY_PWD:/app/fastspeech/model_new  -v LOCAL_TEXT_FILE_PWD:/app/fastspeech/test.txt fastspeech /bin/bash```

In order to start training the model, you need to execute the command inside docker container:  
```python3 train.py```

The ```inference.py``` notebook shows an example of a script that can be used to get wav from text.  

### References
[Repository](https://github.com/markovka17/dla)  
