
# A high-performance HTTP Blaze Model Server.

## How to configure

The server configuration is defined in **serving/frame/predict.proto**. 
It should follow the protobuf text format. 

This config file tells the server which models to load.
For each model, a version name and a dense model file should be specified, 
and a sparse model is optional. 

A example is given in serving/config containing two models, 
one with sparse model, the other without.

## Protocol
The service protocol between client and server is defined by **message Request** and **message Response**, in **serving/frame/predict.proto**.
Notice that instead of TextFormat used in server configure file, the protobuf messages between server and client are transmitted in protobuf **JsonFormat** with HTTP.
This is for clarity since JsonFormat is readable and widely used in HTTP. If you want higher performance, use **BinaryFormat** instead.

## Install and Run
In Blaze root path:

```bash
sh Install.sh
cd serving/
./server config
```
The HTTP server will run on port 8080. Make sure there is no port conflict.

A python example of client is given in **serving/client.py**. 
```bash
python client.py
```
The expected output should be 
```text
{"outputList":[{"name":"output","shape":[2,2],"data":[0.5,0.5,0.5,0.5]}]}
rt: _____ ms
```
The first several request will take relatively long time. 
After the server heats up, the performance will be stably high.


## For higher performance, you can do following optimizations:
1. Replace the HTTP server with other high-performance RPC framework;
2. For protocol between client and server, serialize messages with protobuf binary format instead of JSON string;
3. Create a pool for Predictor for all sessions, rather than create one and release it in each session;
4. Instead of copying numbers one by one, use Memcpy() to write Blaze output to response 
(However, this relies on the implementation of protobuf);
