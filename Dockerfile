FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

COPY Colosseum/PythonClient /app/ColosseumClient

RUN pip install msgpack-rpc-python numpy stable-baselines3 gymnasium \
    -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install /app/ColosseumClient

CMD ["bash"]
