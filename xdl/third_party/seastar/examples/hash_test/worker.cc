#include <assert.h>
#include "common/LogFacade.h"
#include "common/option_parser.h"
#include "common/initializer.h"
#include "common/initializer/constant_initializer.h"
#include "client/client.h"
#include "client/client_wrapper_impl.h"

using namespace ps;

void callBack(const Status& status)
{
    if (!status.IsOk())
    {
        std::cerr << status.ToString() << std::endl;
    }
}

// register dense variable
Status registVariable(client::Client& client, const std::string& variableName)
{
    VariableInfo info;
    info.type = VariableInfo::kIndex;
    info.name = variableName;
    // 400: keys count
    // 8: value length
    info.shape = {400, 8};
    info.datatype = ps::types::DataType::kInt64;
  
    return client.RegisterVariable(variableName, info);
}

// register hash variable
Status registHashVariable(client::Client& client, const std::string& variableName)
{
    VariableInfo info;
    info.type = VariableInfo::kHash;
    info.name = variableName;
    info.shape = {4000, 3};
    info.datatype = ps::types::DataType::kInt64;

    return client.RegisterVariable(variableName, info);
}

// will send variable info to servers
void initVariable(client::Client& client, const std::string& variableName)
{
    Initializer* initializer = new initializer::ConstantInitializer(2);
    std::promise<bool> p;
    client.IndexInitializer(variableName, initializer, [&](const Status& st)
    {
        if (!st.IsOk())
        {
            std::cout << st.ToString() << std::endl;
        }
        p.set_value(true);
    });

    p.get_future().wait();
}

//
void initHashVariable(client::Client& client, const std::string& variableName)
{
    Initializer* initializer = new initializer::ConstantInitializer(2);
    std::promise<bool> p;
    client.HashInitializer(variableName, initializer, [&](const Status& st)
    {
        if (!st.IsOk())
        {
            std::cout << st.ToString() << std::endl;
        }
        p.set_value(true);
    });

    p.get_future().wait();
}

// push dense variable
void densePushVariable(client::Client& client, const std::string& variableName)
{
    Tensor grad(DataType::kInt64, TensorShape({400, 8}), new initializer::ConstantInitializer(2));
    vector<Data*> datas = client.Args(grad);  
    std::promise<bool> p;
    client.DensePush(variableName, "AssignAddUpdater", datas, [&](const Status& st)
    {
        if (!st.IsOk())
        {
            std::cout << st.ToString() << std::endl;
        }
        p.set_value(true);
    });

    p.get_future().wait();
}

// push hash variable
void hashPushVariable(client::Client& client, const std::string& variableName)
{
    // the hash ids,
    // 4: keys count
    // 2: int128 = 2 * int64
    Tensor ids(DataType::kInt64, TensorShape({4, 2}), new initializer::ConstantInitializer(0));
    // fill the real ids
    int64_t* buf = ids.Raw<int64_t>();
    buf[1] = 2;
    buf[3] = 5;
    buf[5] = 199;
    buf[7] = 309;

    // value
    Tensor grad(DataType::kInt64, TensorShape({4, 3}), new initializer::ConstantInitializer(2));
    vector<Data*> datas = client.Args(grad);
    std::promise<bool> p;
    client.HashPush(variableName, ids, "AssignAddUpdater", datas, [&](const Status& st)
    {
        if (!st.IsOk())
        {
            std::cout << st.ToString() << std::endl;
        }
        p.set_value(true);
    });

    p.get_future().wait();
}

// pull dense variable
void densePullVariable(client::Client& client, const std::string& variableName)
{
    Tensor result;
    std::promise<bool> p;
    client.DensePull(variableName, &result, [&](const Status& st)
    {
        if (!st.IsOk())
        {
            std::cout << st.ToString() << std::endl;
        }
        else
        {
            std::cout << "pull result:" << std::endl;
            int64_t* tensorData = result.Raw<int64_t>();
            for (size_t i = 0; i < result.Shape().NumElements(); i++)
            {
                std::cout << tensorData[i] << std::endl;
            }
        }
        p.set_value(true);
    });
    
    p.get_future().wait();
}

// pull hash variable
void hashPullVariable(client::Client& client, const std::string& variableName)
{
    Tensor ids(DataType::kInt64, TensorShape({4, 2}), new initializer::ConstantInitializer(0));
    int64_t* buf = ids.Raw<int64_t>();
    buf[1] = 2;
    buf[3] = 5;
    buf[5] = 199;
    buf[7] = 309;
    Tensor result;
    std::promise<bool> p;
    client.HashPull(variableName, ids, &result, [&](const Status& st)
    {
        if (!st.IsOk())
        {
            std::cout << st.ToString() << std::endl;
        }
        else
        {
            std::cout << "pull result:" << std::endl;
            int64_t* tensorData = result.Raw<int64_t>();
            for (size_t i = 0; i < result.Shape().NumElements(); i++) 
            {
                std::cout << tensorData[i] << std::endl;
            }
        }
        p.set_value(true);
    });
    
    p.get_future().wait();
}


// main
int main(int argc, char** argv)
{
    OptionParser optParser;
    optParser.addOption("-v", "--variable_name", "variable_name", OptionParser::OPT_STRING, true);
    optParser.addOption("-sn", "--server_num", "server_num", OptionParser::OPT_INT32, true);
    optParser.addOption("-sp", "--scheduler_kv_path", "scheduler_kv_path", OptionParser::OPT_STRING, true);
    optParser.addOption("-a", "--action", "action", OptionParser::OPT_STRING, true);
    optParser.addOption("-h", "--hash", "hash", OptionParser::OPT_STRING, true);
    if (!optParser.parseArgs(argc, argv))
    {
        LOG_ERROR("Parse Server Args Error");
        return -1;
    }

    std::string variableName;
    int32_t serverNum;
    std::string schedulerKvPath;
    std::string action;
    std::string useHash;
    optParser.getOptionValue("variable_name", variableName);
    optParser.getOptionValue("server_num", serverNum);
    optParser.getOptionValue("scheduler_kv_path", schedulerKvPath);
    optParser.getOptionValue("action", action);
    optParser.getOptionValue("hash", useHash);
  
    client::ClientArgs args;
    args.scheduler_addr = schedulerKvPath;
    args.client_wrapper_creator = [serverNum](){return new client::ClientWrapperImpl(serverNum);};
    client::RawClient* rawClient = new client::RawClient(args);
    client::Client client(rawClient);
    client.Init();

    if (action == "test")
    {
        if (useHash == "1")
        {
            registHashVariable(client, variableName);
            initHashVariable(client, variableName);
            hashPushVariable(client, variableName);
            hashPullVariable(client, variableName);  
        }
        else
        {
            registVariable(client, variableName);
            initVariable(client, variableName);
            densePushVariable(client, variableName);
            densePullVariable(client, variableName);
        }
  }
  else if (action == "push")
  {
      if (useHash == "1")
      {
          hashPushVariable(client, variableName);
      }
      else
      {
          densePushVariable(client, variableName);
      }
  }
  else
  {
      if (useHash == "1")
      {
          hashPullVariable(client, variableName);
      }
      else
      {
          densePullVariable(client, variableName);
      }
  }

  return 0;
}


