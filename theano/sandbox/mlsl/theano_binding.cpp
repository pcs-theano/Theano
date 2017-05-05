#include <string.h>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/syscall.h>
#include <unistd.h>
#include "mlsl.hpp"

using namespace MLSL;



#define CACHELINE_SIZE 64

#define GET_TID()    syscall(SYS_gettid)
#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#define MLSL_ASSERT(cond, fmt, ...)                                                       \
  do                                                                                      \
  {                                                                                       \
      if (!(cond))                                                                        \
      {                                                                                   \
          fprintf(stderr, "(%ld): %s:%s:%d: ASSERT '%s' FAILED: " fmt "\n",               \
                  GET_TID(), __FILENAME__, __FUNCTION__, __LINE__, #cond, ##__VA_ARGS__); \
          fflush(stderr);                                                                 \
          MLSL::Environment::GetEnv().Finalize();                                         \
          _exit(1);                                                                       \
      }                                                                                   \
  } while(0)


extern "C"
{
    enum mlsl_datatype
    {
        mlsl_float = DT_FLOAT,
        mlsl_double = DT_DOUBLE,
    };

    typedef struct theano_param
    {
        int idx;
        Distribution* dist;
        Operation *op;
        void* comm_buf;
        void* user_buf;
        int buf_size;
    } theano_param;
    int param_count = 0;
    theano_param* params = NULL;

    theano_param& get_param(int idx)
    {
        MLSL_ASSERT(idx >= 1 && idx <= param_count, "idx (%d) should be in [1, %d] range", idx, param_count);
        MLSL_ASSERT(params, "params is null");
        return params[idx - 1];
    }

    bool is_inited = false;
    Session* session = NULL;

    void mlsl_init(int* argc, char** argv[])
    {
        if (!is_inited)
        {
            Environment::GetEnv().Init(argc, argv);
            session = Environment::GetEnv().CreateSession(PT_TRAIN);
            is_inited = true;
        }
    }

    void mlsl_set_global_batch_size(int size)
    {
        MLSL_ASSERT(session, "session is null");
        session->SetGlobalMinibatchSize(size);
    }

    void mlsl_set_param_count(int count)
    {
        param_count = count;
        params = (theano_param*)malloc(sizeof(theano_param) * param_count);
        MLSL_ASSERT(params, "params is null");
    }

    void mlsl_finalize()
    {
        MLSL_ASSERT(session, "session is null");
        if (is_inited)
        {
            for (int idx = 0; idx < param_count; idx++)
            {
                theano_param& param = get_param(idx);
                Environment::GetEnv().Free(param.comm_buf);
                Environment::GetEnv().DeleteDistribution(param.dist);
            }
            Environment::GetEnv().DeleteSession(session);
            Environment::GetEnv().Finalize();

            free(params);
            params = NULL;
            param_count = 0;

            is_inited = false;
        }
    }

    int mlsl_size() { return Environment::GetEnv().GetProcessCount(); }
    int mlsl_rank() { return Environment::GetEnv().GetProcessIdx(); }

    void mlsl_create_distribution(int param_idx)
    {
        theano_param& param = get_param(param_idx);
        param.idx = param_idx;
        param.dist = Environment::GetEnv().CreateDistribution(mlsl_size(), 1);
    }

    void mlsl_create_operation(int param_idx, int buf_count, mlsl_datatype dtype)
    {
        MLSL_ASSERT(session, "session is null");

        theano_param& param = get_param(param_idx);
        OperationRegInfo* reg_info = session->CreateOperationRegInfo(OT_CC);
        std::ostringstream stream;
        stream << param_idx;
        reg_info->SetName(stream.str().c_str());
        reg_info->AddParameterSet(buf_count, 1, (DataType)dtype, false);
        int op_idx = session->AddOperation(reg_info, param.dist);
        param.op = session->GetOperation(op_idx);
        session->DeleteOperationRegInfo(reg_info);
        param.buf_size = buf_count * ((dtype == mlsl_float) ? sizeof(float) : sizeof(double));
        /* FIXME: hide mem copies on MLSL level */
        param.comm_buf = Environment::GetEnv().Alloc(param.buf_size, CACHELINE_SIZE);
    }

    void mlsl_start(int param_idx, void* buf)
    {
        theano_param& param = get_param(param_idx);

        param.user_buf = buf;
        /* FIXME: hide mem copies on MLSL level */
        memcpy(param.comm_buf, param.user_buf, param.buf_size);

        param.op->GetParameterSet(0)->StartGradientComm(param.comm_buf);
    }

    void mlsl_wait(int param_idx)
    {
        theano_param& param = get_param(param_idx);

        void* ret_buf = param.op->GetParameterSet(0)->WaitGradientComm();

        MLSL_ASSERT(ret_buf == param.comm_buf, "different buffers in Start and Wait functions");
        memcpy(param.user_buf, param.comm_buf, param.buf_size);
    }
}
