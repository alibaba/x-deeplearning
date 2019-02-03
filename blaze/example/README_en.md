

# Examples of Blaze C/C++ api & python api

## Location of API source
The C/C++ api is located at **blaze/api**.

The python api is located at **binding/blaze/pyblaze**


## Structure of the API

API of Blaze consist of two main Classes, PredictorManager and Predictor.

Each PredictorManager instance corresponds to one model, 
and each Predictor instance created by this PredictorManager corresponds to one working thread inferring that model.

## Steps to use the API

1. Init PredictorManager and load model
2. Create a Predictor for a working thread
3. Reshape and Feed Inputs to the Predictor
4. Do forward calculation on the Predictor
5. Fetch outputs from the Predictor
6. Release this Predictor instance

A example is given in **example/cpp/predictor_example.cc**

## Profiling

Before Predictor::Forward(), use Predictor::RegisterObservers(observer_names) to register observers.
Supported observers are named "profile" and "cost", corresponding to throughput and latency respectively.
Call Predictor::DumpObservers() after Predictor::Forward() to get the profiling result.

## Debugging
Use Predictor::ListInternalNames() to get the names of internal tensors, 
including intermediate results and constant net parameters. 
Use Predictor::InternalParam() to fetch data of these tensors. 
Also, you can use Predictor::InternalShape() to inspect the shapes.
[](InternalParam()这个函数名有歧义，我以为是要打印模型参数。是否要修改？)

## ATTENTION
In C/C++, Class Predictor / PredictorHandle is **NOT** thread-safe. 
Therefore, users should not access one Predictor instance across different threads at the same time.

Create a new Predictor instance (in form of pointer) from the PredictorManager.
Users handle the ownership of this instance. Remember to release it after fetching outputs. 