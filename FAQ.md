
# FAQ

 - [**CUDA out of memory**](#cuda-out-of-memory)


## CUDA out of memory  

If you encounter an error message like the ones below, it indicates that the GPU memory is insufficient.
```
torch.cuda.OutOfMemoryError: CUDA out of memory.
```

```
Process Keras_worker_ForkPoolWorker-22:
```

To prevent this error, consider reducing the BATCH_SIZE parameter in the TRAINER section of the ini file.
```
[TRAINER]
BATCH_SIZE = 10
```


