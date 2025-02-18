/*
 * File: PolicyNeuralNetwork.cpp
 *
 *
 *   --- THIS FILE GENERATED BY S-FUNCTION BUILDER: 3.0 ---
 *
 *   This file is an S-function produced by the S-Function
 *   Builder which only recognizes certain fields.  Changes made
 *   outside these fields will be lost the next time the block is
 *   used to load, edit, and resave this file. This file will be overwritten
 *   by the S-function Builder block. If you want to edit this file by hand,
 *   you must change it only in the area defined as:
 *
 *        %%%-SFUNWIZ_defines_Changes_BEGIN
 *        #define NAME 'replacement text'
 *        %%% SFUNWIZ_defines_Changes_END
 *
 *   DO NOT change NAME--Change the 'replacement text' only.
 *
 *   For better compatibility with the Simulink Coder, the
 *   "wrapper" S-function technique is used.  This is discussed
 *   in the Simulink Coder's Manual in the Chapter titled,
 *   "Wrapper S-functions".
 *
 *  -------------------------------------------------------------------------
 * | See matlabroot/simulink/src/sfuntmpl_doc.c for a more detailed template |
 *  -------------------------------------------------------------------------
 *
 * Created: Thu Nov 14 21:16:45 2024
 */

#define S_FUNCTION_LEVEL               2
#define S_FUNCTION_NAME                PolicyNeuralNetwork

/*<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<*/
/* %%%-SFUNWIZ_defines_Changes_BEGIN --- EDIT HERE TO _END */
#define NUM_INPUTS                     6

/* Input Port  0 */
#define IN_PORT_0_NAME                 norm_observation
#define INPUT_0_DIMS_ND                {16,1}
#define INPUT_0_NUM_ELEMS              16
#define INPUT_0_WIDTH                  16
#define INPUT_DIMS_0_COL               1
#define INPUT_0_DTYPE                  real32_T
#define INPUT_0_COMPLEX                COMPLEX_NO
#define IN_0_BUS_BASED                 0
#define IN_0_BUS_NAME
#define IN_0_DIMS                      2-D
#define INPUT_0_FEEDTHROUGH            1
#define IN_0_ISSIGNED                  0
#define IN_0_WORDLENGTH                8
#define IN_0_FIXPOINTSCALING           1
#define IN_0_FRACTIONLENGTH            9
#define IN_0_BIAS                      0
#define IN_0_SLOPE                     0.125

/* Input Port  1 */
#define IN_PORT_1_NAME                 norm_observation_size
#define INPUT_1_DIMS_ND                {1,1}
#define INPUT_1_NUM_ELEMS              1
#define INPUT_1_WIDTH                  1
#define INPUT_DIMS_1_COL               1
#define INPUT_1_DTYPE                  uint32_T
#define INPUT_1_COMPLEX                COMPLEX_NO
#define IN_1_BUS_BASED                 0
#define IN_1_BUS_NAME
#define IN_1_DIMS                      2-D
#define INPUT_1_FEEDTHROUGH            1
#define IN_1_ISSIGNED                  0
#define IN_1_WORDLENGTH                8
#define IN_1_FIXPOINTSCALING           1
#define IN_1_FRACTIONLENGTH            9
#define IN_1_BIAS                      0
#define IN_1_SLOPE                     0.125

/* Input Port  2 */
#define IN_PORT_2_NAME                 norm_action_dist_size
#define INPUT_2_DIMS_ND                {1,1}
#define INPUT_2_NUM_ELEMS              1
#define INPUT_2_WIDTH                  1
#define INPUT_DIMS_2_COL               1
#define INPUT_2_DTYPE                  uint32_T
#define INPUT_2_COMPLEX                COMPLEX_NO
#define IN_2_BUS_BASED                 0
#define IN_2_BUS_NAME
#define IN_2_DIMS                      2-D
#define INPUT_2_FEEDTHROUGH            1
#define IN_2_ISSIGNED                  0
#define IN_2_WORDLENGTH                8
#define IN_2_FIXPOINTSCALING           1
#define IN_2_FRACTIONLENGTH            9
#define IN_2_BIAS                      0
#define IN_2_SLOPE                     0.125

/* Input Port  3 */
#define IN_PORT_3_NAME                 static_nn_memory
#define INPUT_3_DIMS_ND                {65536,1}
#define INPUT_3_NUM_ELEMS              65536
#define INPUT_3_WIDTH                  65536
#define INPUT_DIMS_3_COL               1
#define INPUT_3_DTYPE                  uint8_T
#define INPUT_3_COMPLEX                COMPLEX_NO
#define IN_3_BUS_BASED                 0
#define IN_3_BUS_NAME
#define IN_3_DIMS                      2-D
#define INPUT_3_FEEDTHROUGH            1
#define IN_3_ISSIGNED                  0
#define IN_3_WORDLENGTH                8
#define IN_3_FIXPOINTSCALING           1
#define IN_3_FRACTIONLENGTH            9
#define IN_3_BIAS                      0
#define IN_3_SLOPE                     0.125

/* Input Port  4 */
#define IN_PORT_4_NAME                 tensor_arena_size
#define INPUT_4_DIMS_ND                {1,1}
#define INPUT_4_NUM_ELEMS              1
#define INPUT_4_WIDTH                  1
#define INPUT_DIMS_4_COL               1
#define INPUT_4_DTYPE                  uint32_T
#define INPUT_4_COMPLEX                COMPLEX_NO
#define IN_4_BUS_BASED                 0
#define IN_4_BUS_NAME
#define IN_4_DIMS                      1-D
#define INPUT_4_FEEDTHROUGH            1
#define IN_4_ISSIGNED                  0
#define IN_4_WORDLENGTH                8
#define IN_4_FIXPOINTSCALING           1
#define IN_4_FRACTIONLENGTH            9
#define IN_4_BIAS                      0
#define IN_4_SLOPE                     0.125

/* Input Port  5 */
#define IN_PORT_5_NAME                 rl_algorithm
#define INPUT_5_DIMS_ND                {1,1}
#define INPUT_5_NUM_ELEMS              1
#define INPUT_5_WIDTH                  1
#define INPUT_DIMS_5_COL               1
#define INPUT_5_DTYPE                  uint32_T
#define INPUT_5_COMPLEX                COMPLEX_NO
#define IN_5_BUS_BASED                 0
#define IN_5_BUS_NAME
#define IN_5_DIMS                      1-D
#define INPUT_5_FEEDTHROUGH            1
#define IN_5_ISSIGNED                  0
#define IN_5_WORDLENGTH                8
#define IN_5_FIXPOINTSCALING           1
#define IN_5_FRACTIONLENGTH            9
#define IN_5_BIAS                      0
#define IN_5_SLOPE                     0.125
#define NUM_OUTPUTS                    1

/* Output Port  0 */
#define OUT_PORT_0_NAME                norm_action_dist
#define OUTPUT_0_DIMS_ND               {4,1}
#define OUTPUT_0_NUM_ELEMS             4
#define OUTPUT_0_WIDTH                 4
#define OUTPUT_DIMS_0_COL              1
#define OUTPUT_0_DTYPE                 real32_T
#define OUTPUT_0_COMPLEX               COMPLEX_NO
#define OUT_0_BUS_BASED                0
#define OUT_0_BUS_NAME
#define OUT_0_DIMS                     2-D
#define OUT_0_ISSIGNED                 1
#define OUT_0_WORDLENGTH               8
#define OUT_0_FIXPOINTSCALING          1
#define OUT_0_FRACTIONLENGTH           3
#define OUT_0_BIAS                     0
#define OUT_0_SLOPE                    0.125
#define NPARAMS                        0
#define SAMPLE_TIME_0                  INHERITED_SAMPLE_TIME
#define NUM_DISC_STATES                0
#define DISC_STATES_IC                 [0]
#define NUM_CONT_STATES                0
#define CONT_STATES_IC                 [0]
#define SFUNWIZ_GENERATE_TLC           1
#define SOURCEFILES                    "__SFB__tensorflow/lite/micro/kernels/activations.cpp__SFB__tensorflow/lite/micro/kernels/activations_common.cpp__SFB__tensorflow/lite/micro/kernels/add.cpp__SFB__tensorflow/lite/micro/kernels/add_common.cpp__SFB__tensorflow/lite/micro/kernels/add_n.cpp__SFB__tensorflow/lite/micro/kernels/arg_min_max.cpp__SFB__tensorflow/lite/micro/kernels/assign_variable.cpp__SFB__tensorflow/lite/micro/kernels/batch_to_space_nd.cpp__SFB__tensorflow/lite/micro/kernels/broadcast_args.cpp__SFB__tensorflow/lite/micro/kernels/broadcast_to.cpp__SFB__tensorflow/lite/micro/kernels/call_once.cpp__SFB__tensorflow/lite/micro/kernels/cast.cpp__SFB__tensorflow/lite/micro/kernels/ceil.cpp__SFB__tensorflow/lite/micro/kernels/circular_buffer.cpp__SFB__tensorflow/lite/micro/kernels/circular_buffer_common.cpp__SFB__tensorflow/lite/micro/kernels/comparisons.cpp__SFB__tensorflow/lite/micro/kernels/concatenation.cpp__SFB__tensorflow/lite/micro/kernels/conv.cpp__SFB__tensorflow/lite/micro/kernels/conv_common.cpp__SFB__tensorflow/lite/micro/kernels/cumsum.cpp__SFB__tensorflow/lite/micro/kernels/depth_to_space.cpp__SFB__tensorflow/lite/micro/kernels/depthwise_conv.cpp__SFB__tensorflow/lite/micro/kernels/depthwise_conv_common.cpp__SFB__tensorflow/lite/micro/kernels/dequantize.cpp__SFB__tensorflow/lite/micro/kernels/dequantize_common.cpp__SFB__tensorflow/lite/micro/kernels/detection_postprocess.cpp__SFB__tensorflow/lite/micro/kernels/div.cpp__SFB__tensorflow/lite/micro/kernels/elementwise.cpp__SFB__tensorflow/lite/micro/kernels/elu.cpp__SFB__tensorflow/lite/micro/kernels/ethosu.cpp__SFB__tensorflow/lite/micro/kernels/exp.cpp__SFB__tensorflow/lite/micro/kernels/expand_dims.cpp__SFB__tensorflow/lite/micro/kernels/fill.cpp__SFB__tensorflow/lite/micro/kernels/floor.cpp__SFB__tensorflow/lite/micro/kernels/floor_div.cpp__SFB__tensorflow/lite/micro/kernels/floor_mod.cpp__SFB__tensorflow/lite/micro/kernels/fully_connected.cpp__SFB__tensorflow/lite/micro/kernels/fully_connected_common.cpp__SFB__tensorflow/lite/micro/kernels/gather.cpp__SFB__tensorflow/lite/micro/kernels/gather_nd.cpp__SFB__tensorflow/lite/micro/kernels/hard_swish.cpp__SFB__tensorflow/lite/micro/kernels/hard_swish_common.cpp__SFB__tensorflow/lite/micro/kernels/if.cpp__SFB__tensorflow/lite/micro/kernels/kernel_runner.cpp__SFB__tensorflow/lite/micro/kernels/kernel_util_micro.cpp__SFB__tensorflow/lite/micro/kernels/l2norm.cpp__SFB__tensorflow/lite/micro/kernels/l2_pool_2d.cpp__SFB__tensorflow/lite/micro/kernels/leaky_relu.cpp__SFB__tensorflow/lite/micro/kernels/leaky_relu_common.cpp__SFB__tensorflow/lite/micro/kernels/logical.cpp__SFB__tensorflow/lite/micro/kernels/logical_common.cpp__SFB__tensorflow/lite/micro/kernels/logistic.cpp__SFB__tensorflow/lite/micro/kernels/logistic_common.cpp__SFB__tensorflow/lite/micro/kernels/log_softmax.cpp__SFB__tensorflow/lite/micro/kernels/lstm_eval.cpp__SFB__tensorflow/lite/micro/kernels/maximum_minimum.cpp__SFB__tensorflow/lite/micro/kernels/micro_tensor_utils.cpp__SFB__tensorflow/lite/micro/kernels/mirror_pad.cpp__SFB__tensorflow/lite/micro/kernels/mul.cpp__SFB__tensorflow/lite/micro/kernels/mul_common.cpp__SFB__tensorflow/lite/micro/kernels/neg.cpp__SFB__tensorflow/lite/micro/kernels/pack.cpp__SFB__tensorflow/lite/micro/kernels/pad.cpp__SFB__tensorflow/lite/micro/kernels/pooling.cpp__SFB__tensorflow/lite/micro/kernels/pooling_common.cpp__SFB__tensorflow/lite/micro/kernels/prelu.cpp__SFB__tensorflow/lite/micro/kernels/prelu_common.cpp__SFB__tensorflow/lite/micro/kernels/quantize.cpp__SFB__tensorflow/lite/micro/kernels/quantize_common.cpp__SFB__tensorflow/lite/micro/kernels/read_variable.cpp__SFB__tensorflow/lite/micro/kernels/reduce.cpp__SFB__tensorflow/lite/micro/kernels/reduce_common.cpp__SFB__tensorflow/lite/micro/kernels/reshape.cpp__SFB__tensorflow/lite/micro/kernels/resize_bilinear.cpp__SFB__tensorflow/lite/micro/kernels/resize_nearest_neighbor.cpp__SFB__tensorflow/lite/micro/kernels/round.cpp__SFB__tensorflow/lite/micro/kernels/shape.cpp__SFB__tensorflow/lite/micro/kernels/slice.cpp__SFB__tensorflow/lite/micro/kernels/softmax.cpp__SFB__tensorflow/lite/micro/kernels/softmax_common.cpp__SFB__tensorflow/lite/micro/kernels/space_to_batch_nd.cpp__SFB__tensorflow/lite/micro/kernels/space_to_depth.cpp__SFB__tensorflow/lite/micro/kernels/split.cpp__SFB__tensorflow/lite/micro/kernels/split_v.cpp__SFB__tensorflow/lite/micro/kernels/squared_difference.cpp__SFB__tensorflow/lite/micro/kernels/squeeze.cpp__SFB__tensorflow/lite/micro/kernels/strided_slice.cpp__SFB__tensorflow/lite/micro/kernels/sub.cpp__SFB__tensorflow/lite/micro/kernels/sub_common.cpp__SFB__tensorflow/lite/micro/kernels/svdf.cpp__SFB__tensorflow/lite/micro/kernels/svdf_common.cpp__SFB__tensorflow/lite/micro/kernels/tanh.cpp__SFB__tensorflow/lite/micro/kernels/transpose.cpp__SFB__tensorflow/lite/micro/kernels/transpose_conv.cpp__SFB__tensorflow/lite/micro/kernels/unidirectional_sequence_lstm.cpp__SFB__tensorflow/lite/micro/kernels/unpack.cpp__SFB__tensorflow/lite/micro/kernels/var_handle.cpp__SFB__tensorflow/lite/micro/kernels/while.cpp__SFB__tensorflow/lite/micro/kernels/zeros_like.cpp__SFB__tensorflow/lite/core/api/flatbuffer_conversions.cpp__SFB__tensorflow/lite/core/api/op_resolver.cpp__SFB__tensorflow/lite/core/api/tensor_utils.cpp__SFB__tensorflow/lite/core/api/error_reporter.cpp__SFB__tensorflow/lite/schema/schema_utils.cpp__SFB__tensorflow/lite/kernels/kernel_util.cpp__SFB__tensorflow/lite/kernels/internal/reference/portable_tensor_utils.cpp__SFB__tensorflow/lite/kernels/internal/quantization_util.cpp__SFB__tensorflow/lite/c/common.cpp__SFB__tensorflow/lite/micro/memory_helpers.cpp__SFB__tensorflow/lite/micro/all_ops_resolver.cpp__SFB__tensorflow/lite/micro/micro_graph.cpp__SFB__tensorflow/lite/micro/micro_error_reporter.cpp__SFB__tensorflow/lite/micro/micro_context.cpp__SFB__tensorflow/lite/micro/micro_profiler.cpp__SFB__tensorflow/lite/micro/test_helpers.cpp__SFB__tensorflow/lite/micro/recording_micro_allocator.cpp__SFB__tensorflow/lite/micro/micro_interpreter.cpp__SFB__tensorflow/lite/micro/micro_time.cpp__SFB__tensorflow/lite/micro/micro_resource_variable.cpp__SFB__tensorflow/lite/micro/test_helper_custom_ops.cpp__SFB__tensorflow/lite/micro/flatbuffer_utils.cpp__SFB__tensorflow/lite/micro/mock_micro_graph.cpp__SFB__tensorflow/lite/micro/debug_log.cpp__SFB__tensorflow/lite/micro/micro_allocation_info.cpp__SFB__tensorflow/lite/micro/system_setup.cpp__SFB__tensorflow/lite/micro/micro_allocator.cpp__SFB__tensorflow/lite/micro/micro_string.cpp__SFB__tensorflow/lite/micro/micro_utils.cpp__SFB__tensorflow/lite/micro/fake_micro_context.cpp__SFB__tensorflow/lite/micro/arena_allocator/persistent_arena_buffer_allocator.cpp__SFB__tensorflow/lite/micro/arena_allocator/recording_single_arena_buffer_allocator.cpp__SFB__tensorflow/lite/micro/arena_allocator/non_persistent_arena_buffer_allocator.cpp__SFB__tensorflow/lite/micro/arena_allocator/single_arena_buffer_allocator.cpp__SFB__tensorflow/lite/micro/memory_planner/greedy_memory_planner.cpp__SFB__tensorflow/lite/micro/memory_planner/linear_memory_planner.cpp__SFB__tensorflow/lite/micro/memory_planner/non_persistent_buffer_planner_shim.cpp"
#define PANELINDEX                     N/A
#define USE_SIMSTRUCT                  0
#define SHOW_COMPILE_STEPS             0
#define CREATE_DEBUG_MEXFILE           0
#define SAVE_CODE_ONLY                 0
#define SFUNWIZ_REVISION               3.0

/* %%%-SFUNWIZ_defines_Changes_END --- EDIT HERE TO _BEGIN */
/*<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<*/
#include "simstruc.h"

extern void PolicyNeuralNetwork_Outputs_wrapper(const real32_T *norm_observation,
  const uint32_T *norm_observation_size,
  const uint32_T *norm_action_dist_size,
  const uint8_T *static_nn_memory,
  const uint32_T *tensor_arena_size,
  const uint32_T *rl_algorithm,
  real32_T *norm_action_dist);

/*====================*
 * S-function methods *
 *====================*/
/* Function: mdlInitializeSizes ===============================================
 * Abstract:
 *   Setup sizes of the various vectors.
 */
static void mdlInitializeSizes(SimStruct *S)
{
  DECL_AND_INIT_DIMSINFO(inputDimsInfo);
  DECL_AND_INIT_DIMSINFO(outputDimsInfo);
  ssSetNumSFcnParams(S, NPARAMS);
  if (ssGetNumSFcnParams(S) != ssGetSFcnParamsCount(S)) {
    return;                            /* Parameter mismatch will be reported by Simulink */
  }

  ssSetArrayLayoutForCodeGen(S, SS_COLUMN_MAJOR);
  ssSetOperatingPointCompliance(S, USE_DEFAULT_OPERATING_POINT);
  ssSetNumContStates(S, NUM_CONT_STATES);
  ssSetNumDiscStates(S, NUM_DISC_STATES);
  if (!ssSetNumInputPorts(S, NUM_INPUTS))
    return;

  /* Input Port 0 */
  ssAllowSignalsWithMoreThan2D(S);
  inputDimsInfo.numDims = 2;
  inputDimsInfo.width = INPUT_0_NUM_ELEMS;
  int_T in0Dims[] = INPUT_0_DIMS_ND;
  inputDimsInfo.dims = in0Dims;
  ssSetInputPortDimensionInfo(S, 0, &inputDimsInfo);
  ssSetInputPortDataType(S, 0, SS_SINGLE);
  ssSetInputPortComplexSignal(S, 0, INPUT_0_COMPLEX);
  ssSetInputPortDirectFeedThrough(S, 0, INPUT_0_FEEDTHROUGH);
  ssSetInputPortRequiredContiguous(S, 0, 1);/*direct input signal access*/

  /* Input Port 1 */
  ssSetInputPortWidth(S, 1, INPUT_1_NUM_ELEMS);
  ssSetInputPortDataType(S, 1, SS_UINT32);
  ssSetInputPortComplexSignal(S, 1, INPUT_1_COMPLEX);
  ssSetInputPortDirectFeedThrough(S, 1, INPUT_1_FEEDTHROUGH);
  ssSetInputPortRequiredContiguous(S, 1, 1);/*direct input signal access*/

  /* Input Port 2 */
  ssSetInputPortWidth(S, 2, INPUT_2_NUM_ELEMS);
  ssSetInputPortDataType(S, 2, SS_UINT32);
  ssSetInputPortComplexSignal(S, 2, INPUT_2_COMPLEX);
  ssSetInputPortDirectFeedThrough(S, 2, INPUT_2_FEEDTHROUGH);
  ssSetInputPortRequiredContiguous(S, 2, 1);/*direct input signal access*/

  /* Input Port 3 */
  inputDimsInfo.numDims = 2;
  inputDimsInfo.width = INPUT_3_NUM_ELEMS;
  int_T in3Dims[] = INPUT_3_DIMS_ND;
  inputDimsInfo.dims = in3Dims;
  ssSetInputPortDimensionInfo(S, 3, &inputDimsInfo);
  ssSetInputPortDataType(S, 3, SS_UINT8);
  ssSetInputPortComplexSignal(S, 3, INPUT_3_COMPLEX);
  ssSetInputPortDirectFeedThrough(S, 3, INPUT_3_FEEDTHROUGH);
  ssSetInputPortRequiredContiguous(S, 3, 1);/*direct input signal access*/

  /* Input Port 4 */
  ssSetInputPortWidth(S, 4, INPUT_4_NUM_ELEMS);
  ssSetInputPortDataType(S, 4, SS_UINT32);
  ssSetInputPortComplexSignal(S, 4, INPUT_4_COMPLEX);
  ssSetInputPortDirectFeedThrough(S, 4, INPUT_4_FEEDTHROUGH);
  ssSetInputPortRequiredContiguous(S, 4, 1);/*direct input signal access*/

  /* Input Port 5 */
  ssSetInputPortWidth(S, 5, INPUT_5_NUM_ELEMS);
  ssSetInputPortDataType(S, 5, SS_UINT32);
  ssSetInputPortComplexSignal(S, 5, INPUT_5_COMPLEX);
  ssSetInputPortDirectFeedThrough(S, 5, INPUT_5_FEEDTHROUGH);
  ssSetInputPortRequiredContiguous(S, 5, 1);/*direct input signal access*/
  if (!ssSetNumOutputPorts(S, NUM_OUTPUTS))
    return;

  /* Output Port 0 */
  outputDimsInfo.numDims = 2;
  outputDimsInfo.width = OUTPUT_0_NUM_ELEMS;
  int_T out0Dims[] = OUTPUT_0_DIMS_ND;
  outputDimsInfo.dims = out0Dims;
  ssSetOutputPortDimensionInfo(S, 0, &outputDimsInfo);
  ssSetOutputPortDataType(S, 0, SS_SINGLE);
  ssSetOutputPortComplexSignal(S, 0, OUTPUT_0_COMPLEX);
  ssSetNumPWork(S, 0);
  ssSetNumSampleTimes(S, 1);
  ssSetNumRWork(S, 0);
  ssSetNumIWork(S, 0);
  ssSetNumModes(S, 0);
  ssSetNumNonsampledZCs(S, 0);
  ssSetSimulinkVersionGeneratedIn(S, "10.7");

  /* Take care when specifying exception free code - see sfuntmpl_doc.c */
  ssSetOptions(S, (SS_OPTION_EXCEPTION_FREE_CODE |
                   SS_OPTION_USE_TLC_WITH_ACCELERATOR |
                   SS_OPTION_WORKS_WITH_CODE_REUSE));
}

#if defined(MATLAB_MEX_FILE)
#define MDL_SET_INPUT_PORT_DIMENSION_INFO

static void mdlSetInputPortDimensionInfo(SimStruct *S,
  int_T port,
  const DimsInfo_T *dimsInfo)
{
  if (!ssSetInputPortDimensionInfo(S, port, dimsInfo))
    return;
}

#endif

#define MDL_SET_OUTPUT_PORT_DIMENSION_INFO
#if defined(MDL_SET_OUTPUT_PORT_DIMENSION_INFO)

static void mdlSetOutputPortDimensionInfo(SimStruct *S,
  int_T port,
  const DimsInfo_T *dimsInfo)
{
  if (!ssSetOutputPortDimensionInfo(S, port, dimsInfo))
    return;
}

#endif

#define MDL_SET_DEFAULT_PORT_DIMENSION_INFO

static void mdlSetDefaultPortDimensionInfo(SimStruct *S)
{
  DECL_AND_INIT_DIMSINFO(portDimsInfo);
  int_T dims[2];

  /* Setting default dimensions for input port 0 */
  portDimsInfo.width = INPUT_0_NUM_ELEMS;
  dims[0] = INPUT_0_NUM_ELEMS;
  dims[1] = 1;
  portDimsInfo.numDims = 2;
  if (ssGetInputPortWidth(S, 0) == DYNAMICALLY_SIZED) {
    ssSetInputPortMatrixDimensions(S, 0, 1 , 1);
  }

  /* Setting default dimensions for input port 3 */
  portDimsInfo.width = INPUT_3_NUM_ELEMS;
  dims[0] = INPUT_3_NUM_ELEMS;
  dims[1] = 1;
  portDimsInfo.numDims = 2;
  if (ssGetInputPortWidth(S, 3) == DYNAMICALLY_SIZED) {
    ssSetInputPortMatrixDimensions(S, 3, 1 , 1);
  }

  /* Setting default dimensions for output port 0 */
  portDimsInfo.width = OUTPUT_0_NUM_ELEMS;
  dims[0] = OUTPUT_0_NUM_ELEMS;
  dims[1] = 1;
  portDimsInfo.numDims = 2;
  if (ssGetOutputPortNumDimensions(S, 0) == (-1)) {
    ssSetOutputPortDimensionInfo(S, 0, &portDimsInfo);
  }

  return;
}

/* Function: mdlInitializeSampleTimes =========================================
 * Abstract:
 *    Specifiy  the sample time.
 */
static void mdlInitializeSampleTimes(SimStruct *S)
{
  ssSetSampleTime(S, 0, SAMPLE_TIME_0);
  ssSetModelReferenceSampleTimeDefaultInheritance(S);
  ssSetOffsetTime(S, 0, 0.0);
}

#define MDL_SET_INPUT_PORT_DATA_TYPE

static void mdlSetInputPortDataType(SimStruct *S, int port, DTypeId dType)
{
  ssSetInputPortDataType(S, 0, dType);
}

#define MDL_SET_OUTPUT_PORT_DATA_TYPE

static void mdlSetOutputPortDataType(SimStruct *S, int port, DTypeId dType)
{
  ssSetOutputPortDataType(S, 0, dType);
}

#define MDL_SET_DEFAULT_PORT_DATA_TYPES

static void mdlSetDefaultPortDataTypes(SimStruct *S)
{
  ssSetInputPortDataType(S, 0, SS_DOUBLE);
  ssSetOutputPortDataType(S, 0, SS_DOUBLE);
}

#define MDL_START                                                /* Change to #undef to remove function */
#if defined(MDL_START)

/* Function: mdlStart =======================================================
 * Abstract:
 *    This function is called once at start of model execution. If you
 *    have states that should be initialized once, this is the place
 *    to do it.
 */
static void mdlStart(SimStruct *S)
{
}

#endif                                 /*  MDL_START */

/* Function: mdlOutputs =======================================================
 *
 */
static void mdlOutputs(SimStruct *S, int_T tid)
{
  const real32_T *norm_observation = (real32_T *) ssGetInputPortRealSignal(S, 0);
  const uint32_T *norm_observation_size = (uint32_T *) ssGetInputPortRealSignal
    (S, 1);
  const uint32_T *norm_action_dist_size = (uint32_T *) ssGetInputPortRealSignal
    (S, 2);
  const uint8_T *static_nn_memory = (uint8_T *) ssGetInputPortRealSignal(S, 3);
  const uint32_T *tensor_arena_size = (uint32_T *) ssGetInputPortRealSignal(S, 4);
  const uint32_T *rl_algorithm = (uint32_T *) ssGetInputPortRealSignal(S, 5);
  real32_T *norm_action_dist = (real32_T *) ssGetOutputPortRealSignal(S, 0);
  PolicyNeuralNetwork_Outputs_wrapper(norm_observation, norm_observation_size,
    norm_action_dist_size, static_nn_memory, tensor_arena_size, rl_algorithm,
    norm_action_dist);
}

/* Function: mdlTerminate =====================================================
 * Abstract:
 *    In this function, you should perform any actions that are necessary
 *    at the termination of a simulation.  For example, if memory was
 *    allocated in mdlStart, this is the place to free it.
 */
static void mdlTerminate(SimStruct *S)
{
}

#ifdef MATLAB_MEX_FILE                 /* Is this file being compiled as a MEX-file? */
#include "simulink.c"                  /* MEX-file interface mechanism */
#else
#include "cg_sfun.h"                   /* Code generation registration function */
#endif
