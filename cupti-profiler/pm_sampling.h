//
// Copyright 2024 NVIDIA Corporation. All rights reserved
//

#pragma once

// System headers
#include <fstream>
#include <iomanip>
#include <iostream>
#include <unordered_map>
#include <vector>

// CUPTI headers
#include "helper_cupti.h"
#include <cupti_pmsampling.h>
#include <cupti_profiler_host.h>
#include <cupti_profiler_target.h>
#include <cupti_target.h>

/**
 * @brief Represents a single sampling range with associated metrics.
 *
 * This structure holds the data for a single sampling period, including
 * start and end timestamps, and the values of all measured metrics.
 */
struct SamplerRange {
  size_t rangeIndex;
  uint64_t startTimestamp;
  uint64_t endTimestamp;
  std::unordered_map<std::string, double> metricValues;
};

/**
 * @brief Host-side profiler interface for CUPTI PM sampling.
 *
 * This class manages the host-side operations for performance metric sampling,
 * including metric configuration, data collection, and evaluation.
 */
class CuptiProfilerHost {
  std::string m_chipName;
  std::vector<SamplerRange> m_samplerRanges;
  CUpti_Profiler_Host_Object *m_pHostObject = nullptr;

public:
  CuptiProfilerHost() = default;
  ~CuptiProfilerHost() = default;

  /**
   * @brief Get the underlying CUPTI profiler host object.
   * @return Pointer to the CUPTI profiler host object.
   */
  CUpti_Profiler_Host_Object *GetHostObject() { return m_pHostObject; }

  /**
   * @brief Initialize the profiler host with chip and counter availability
   * information.
   * @param chipName Name of the GPU chip being profiled.
   * @param counterAvailibilityImage Buffer containing counter availability
   * data.
   */
  void SetUp(std::string chipName,
             std::vector<uint8_t> &counterAvailibilityImage) {
    m_chipName = chipName;
    CUPTI_API_CALL(Initialize(counterAvailibilityImage));
  }

  /**
   * @brief Clean up profiler host resources.
   */
  void TearDown() { CUPTI_API_CALL(Deinitialize()); }

  /**
   * @brief Create a configuration image for the specified metrics.
   * @param metricsList List of metrics to configure.
   * @param configImage Output buffer for the configuration data.
   * @return CUPTI result status.
   */
  CUptiResult CreateConfigImage(std::vector<const char *> metricsList,
                                std::vector<uint8_t> &configImage) {
    // Add metrics to config image
    {
      CUpti_Profiler_Host_ConfigAddMetrics_Params configAddMetricsParams{
          CUpti_Profiler_Host_ConfigAddMetrics_Params_STRUCT_SIZE};
      configAddMetricsParams.pHostObject = m_pHostObject;
      configAddMetricsParams.ppMetricNames = metricsList.data();
      configAddMetricsParams.numMetrics = metricsList.size();
      CUPTI_API_CALL(
          cuptiProfilerHostConfigAddMetrics(&configAddMetricsParams));
    }

    // Get Config image size and data
    {
      CUpti_Profiler_Host_GetConfigImageSize_Params getConfigImageSizeParams{
          CUpti_Profiler_Host_GetConfigImageSize_Params_STRUCT_SIZE};
      getConfigImageSizeParams.pHostObject = m_pHostObject;
      CUPTI_API_CALL(
          cuptiProfilerHostGetConfigImageSize(&getConfigImageSizeParams));
      configImage.resize(getConfigImageSizeParams.configImageSize);

      CUpti_Profiler_Host_GetConfigImage_Params getConfigImageParams = {
          CUpti_Profiler_Host_GetConfigImage_Params_STRUCT_SIZE};
      getConfigImageParams.pHostObject = m_pHostObject;
      getConfigImageParams.pConfigImage = configImage.data();
      getConfigImageParams.configImageSize = configImage.size();
      CUPTI_API_CALL(cuptiProfilerHostGetConfigImage(&getConfigImageParams));
    }

    // Get Num of Passes
    {
      CUpti_Profiler_Host_GetNumOfPasses_Params getNumOfPassesParam{
          CUpti_Profiler_Host_GetNumOfPasses_Params_STRUCT_SIZE};
      getNumOfPassesParam.pConfigImage = configImage.data();
      getNumOfPassesParam.configImageSize = configImage.size();
      CUPTI_API_CALL(cuptiProfilerHostGetNumOfPasses(&getNumOfPassesParam));
      std::cout << "Num of Passes: " << getNumOfPassesParam.numOfPasses << "\n";
    }

    return CUPTI_SUCCESS;
  }

  /**
   * @brief Evaluate counter data for a sampling range.
   * @param pSamplingObject PM sampling object.
   * @param rangeIndex Index of the range to evaluate.
   * @param metricsList List of metrics to evaluate.
   * @param counterDataImage Buffer containing counter data.
   * @return CUPTI result status.
   */
  CUptiResult EvaluateCounterData(CUpti_PmSampling_Object *pSamplingObject,
                                  size_t rangeIndex,
                                  std::vector<const char *> metricsList,
                                  std::vector<uint8_t> &counterDataImage) {
    m_samplerRanges.push_back(SamplerRange{});
    SamplerRange &samplerRange = m_samplerRanges.back();

    CUpti_PmSampling_CounterData_GetSampleInfo_Params getSampleInfoParams = {
        CUpti_PmSampling_CounterData_GetSampleInfo_Params_STRUCT_SIZE};
    getSampleInfoParams.pPmSamplingObject = pSamplingObject;
    getSampleInfoParams.pCounterDataImage = counterDataImage.data();
    getSampleInfoParams.counterDataImageSize = counterDataImage.size();
    getSampleInfoParams.sampleIndex = rangeIndex;
    CUPTI_API_CALL(
        cuptiPmSamplingCounterDataGetSampleInfo(&getSampleInfoParams));

    samplerRange.startTimestamp = getSampleInfoParams.startTimestamp;
    samplerRange.endTimestamp = getSampleInfoParams.endTimestamp;

    std::vector<double> metricValues(metricsList.size());
    CUpti_Profiler_Host_EvaluateToGpuValues_Params evalauateToGpuValuesParams{
        CUpti_Profiler_Host_EvaluateToGpuValues_Params_STRUCT_SIZE};
    evalauateToGpuValuesParams.pHostObject = m_pHostObject;
    evalauateToGpuValuesParams.pCounterDataImage = counterDataImage.data();
    evalauateToGpuValuesParams.counterDataImageSize = counterDataImage.size();
    evalauateToGpuValuesParams.ppMetricNames = metricsList.data();
    evalauateToGpuValuesParams.numMetrics = metricsList.size();
    evalauateToGpuValuesParams.rangeIndex = rangeIndex;
    evalauateToGpuValuesParams.pMetricValues = metricValues.data();
    CUPTI_API_CALL(
        cuptiProfilerHostEvaluateToGpuValues(&evalauateToGpuValuesParams));

    for (size_t i = 0; i < metricsList.size(); ++i) {
      samplerRange.metricValues[metricsList[i]] = metricValues[i];
    }

    return CUPTI_SUCCESS;
  }

  /**
   * @brief Print sampling ranges and their metric values.
   * Outputs the first 50 samples with their timestamps and metric values.
   */
  void PrintSampleRanges() {
    std::cout << "Total num of Samples: " << m_samplerRanges.size() << "\n";
    std::cout << "Printing first 50 samples:" << "\n";
    for (size_t sampleIndex = 0; sampleIndex < 50; ++sampleIndex) {
      const auto &samplerRange = m_samplerRanges[sampleIndex];
      std::cout << "Sample Index: " << sampleIndex << "\n";
      std::cout << "Timestamps -> Start: [" << samplerRange.startTimestamp
                << "] \tEnd: [" << samplerRange.endTimestamp << "]" << "\n";
      std::cout << "-----------------------------------------------------------"
                   "------------------------\n";
      for (const auto &metric : samplerRange.metricValues) {
        std::cout << std::fixed << std::setprecision(3);
        std::cout << std::setw(50) << std::left << metric.first;
        std::cout << std::setw(30) << std::right << metric.second << "\n";
      }
      std::cout << "-----------------------------------------------------------"
                   "------------------------\n\n";
    }
  }

  /**
   * @brief Get list of supported base metrics.
   * @param metricsList Output vector to store metric names.
   * @return CUPTI result status.
   */
  CUptiResult GetSupportedBaseMetrics(std::vector<std::string> &metricsList) {
    for (size_t metricTypeIndex = 0; metricTypeIndex < CUPTI_METRIC_TYPE__COUNT;
         ++metricTypeIndex) {
      CUpti_Profiler_Host_GetBaseMetrics_Params getBaseMetricsParams{
          CUpti_Profiler_Host_GetBaseMetrics_Params_STRUCT_SIZE};
      getBaseMetricsParams.pHostObject = m_pHostObject;
      getBaseMetricsParams.metricType = (CUpti_MetricType)metricTypeIndex;
      CUPTI_API_CALL(cuptiProfilerHostGetBaseMetrics(&getBaseMetricsParams));

      for (size_t metricIndex = 0;
           metricIndex < getBaseMetricsParams.numMetrics; ++metricIndex) {
        metricsList.push_back(getBaseMetricsParams.ppMetricNames[metricIndex]);
      }
    }
    return CUPTI_SUCCESS;
  }

  /**
   * @brief Get properties of a specific metric.
   * @param metricName Name of the metric.
   * @param metricType Output parameter for metric type.
   * @param metricDescription Output parameter for metric description.
   * @return CUPTI result status.
   */
  CUptiResult GetMetricProperties(const std::string &metricName,
                                  CUpti_MetricType &metricType,
                                  std::string &metricDescription) {
    CUpti_Profiler_Host_GetMetricProperties_Params getMetricPropertiesParams{
        CUpti_Profiler_Host_GetMetricProperties_Params_STRUCT_SIZE};
    getMetricPropertiesParams.pHostObject = m_pHostObject;
    getMetricPropertiesParams.pMetricName = metricName.c_str();

    CUptiResult result = cuptiProfilerHostGetMetricProperties(&getMetricPropertiesParams);
    if (result == CUPTI_SUCCESS) {
      metricType = getMetricPropertiesParams.metricType;
      metricDescription = getMetricPropertiesParams.pDescription;
    }
    return result;
  }

  /**
   * @brief Determine metric type by testing different suffixes
   * @param metricName Base name of the metric to test
   * @return CUpti_MetricType indicating the determined type
   */
  std::string GetMetricsType(const std::string& metricName) {
    CUpti_MetricType metricType;
    std::string description;

    // Try Counter suffix
    if (GetMetricProperties(metricName + ".sum", metricType, description) == CUPTI_SUCCESS) {
      return "Counter";
    }

    // Try Ratio suffix
    if (GetMetricProperties(metricName + ".ratio", metricType, description) == CUPTI_SUCCESS) {
      return "Ratio";
    }

    // Try Throughput suffix
    if (GetMetricProperties(metricName + ".avg.pct_of_peak_sustained_elapsed", metricType, description) == CUPTI_SUCCESS) {
      return "Throughput";
    }

    // If no suffix worked, try getting properties of base metric
    if (GetMetricProperties(metricName, metricType, description) == CUPTI_SUCCESS) {
      switch (metricType) {
        case CUPTI_METRIC_TYPE_COUNTER:
          return "Counter";
        case CUPTI_METRIC_TYPE_RATIO:
          return "Ratio";
        case CUPTI_METRIC_TYPE_THROUGHPUT:
          return "Throughput";
        default:
          return "Unknown";
      }
    }

    // Default case if metric type cannot be determined
    return "Unknown";
  }

  /**
   * @brief Get sub-metrics for a given metric.
   * @param metricName Name of the parent metric.
   * @param subMetricsList Output vector to store sub-metric names.
   * @return CUPTI result status.
   */
  CUptiResult GetSubMetrics(const std::string &metricName,
                            std::vector<std::string> &subMetricsList) {
    CUpti_MetricType metricType;
    std::string metricDescription;
    CUPTI_API_CALL(
        GetMetricProperties(metricName, metricType, metricDescription));

    CUpti_Profiler_Host_GetSubMetrics_Params getSubMetricsParams{
        CUpti_Profiler_Host_GetSubMetrics_Params_STRUCT_SIZE};
    getSubMetricsParams.pHostObject = m_pHostObject;
    getSubMetricsParams.pMetricName = metricName.c_str();
    getSubMetricsParams.metricType = metricType;
    CUPTI_API_CALL(cuptiProfilerHostGetSubMetrics(&getSubMetricsParams));

    for (size_t subMetricIndex = 0;
         subMetricIndex < getSubMetricsParams.numOfSubmetrics;
         ++subMetricIndex) {
      subMetricsList.push_back(
          getSubMetricsParams.ppSubMetrics[subMetricIndex]);
    }
    return CUPTI_SUCCESS;
  }

private:
  CUptiResult Initialize(std::vector<uint8_t> &counterAvailibilityImage) {
    CUpti_Profiler_Host_Initialize_Params hostInitializeParams = {
        CUpti_Profiler_Host_Initialize_Params_STRUCT_SIZE};
    hostInitializeParams.profilerType = CUPTI_PROFILER_TYPE_PM_SAMPLING;
    hostInitializeParams.pChipName = m_chipName.c_str();
    hostInitializeParams.pCounterAvailabilityImage =
        counterAvailibilityImage.data();
    CUPTI_API_CALL(cuptiProfilerHostInitialize(&hostInitializeParams));
    m_pHostObject = hostInitializeParams.pHostObject;
    return CUPTI_SUCCESS;
  }

  CUptiResult Deinitialize() {
    CUpti_Profiler_Host_Deinitialize_Params deinitializeParams = {
        CUpti_Profiler_Host_Deinitialize_Params_STRUCT_SIZE};
    deinitializeParams.pHostObject = m_pHostObject;
    CUPTI_API_CALL(cuptiProfilerHostDeinitialize(&deinitializeParams));
    return CUPTI_SUCCESS;
  }
};

/**
 * @brief Device-side PM sampling interface.
 *
 * This class manages the device-side operations for performance metric
 * sampling, including sampling configuration, control, and data collection.
 */
class CuptiPmSampling {
  CUpti_PmSampling_Object *m_pmSamplerObject = nullptr;

public:
  CuptiPmSampling() = default;
  ~CuptiPmSampling() = default;

  /**
   * @brief Initialize PM sampling for a CUDA device.
   * @param deviceIndex Index of the CUDA device.
   */
  void SetUp(int deviceIndex) {
    std::cout << "CUDA Device Number: " << deviceIndex << "\n";
    CUdevice cuDevice;
    DRIVER_API_CALL(cuDeviceGet(&cuDevice, deviceIndex));

    {
      int computeCapabilityMajor = 0, computeCapabilityMinor = 0;
      DRIVER_API_CALL(cuDeviceGetAttribute(
          &computeCapabilityMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
          cuDevice));
      DRIVER_API_CALL(cuDeviceGetAttribute(
          &computeCapabilityMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
          cuDevice));
      std::cout << "Compute Capability of Device: " << computeCapabilityMajor
                << "." << computeCapabilityMinor << "\n";
      if (computeCapabilityMajor * 10 + computeCapabilityMinor < 75) {
        std::cerr << "Sample not supported as it requires compute capability "
                     ">= 7.5\n";
        exit(0);
      }
    }

    CUPTI_API_CALL(InitializeProfiler());
  }

  /**
   * @brief Clean up PM sampling resources.
   */
  void TearDown() { CUPTI_API_CALL(DeInitializeProfiler()); }

  /**
   * @brief Create a counter data image for storing samples.
   * @param maxSamples Maximum number of samples to store.
   * @param metricsList List of metrics to sample.
   * @param counterDataImage Output buffer for counter data.
   * @return CUPTI result status.
   */
  CUptiResult CreateCounterDataImage(uint64_t maxSamples,
                                     std::vector<const char *> metricsList,
                                     std::vector<uint8_t> &counterDataImage) {
    CUpti_PmSampling_GetCounterDataSize_Params getCounterDataSizeParams = {
        CUpti_PmSampling_GetCounterDataSize_Params_STRUCT_SIZE};
    getCounterDataSizeParams.pPmSamplingObject = m_pmSamplerObject;
    getCounterDataSizeParams.numMetrics = metricsList.size();
    getCounterDataSizeParams.pMetricNames = metricsList.data();
    getCounterDataSizeParams.maxSamples = maxSamples;
    CUPTI_API_CALL(
        cuptiPmSamplingGetCounterDataSize(&getCounterDataSizeParams));

    counterDataImage.resize(getCounterDataSizeParams.counterDataSize);
    CUpti_PmSampling_CounterDataImage_Initialize_Params initializeParams{
        CUpti_PmSampling_CounterDataImage_Initialize_Params_STRUCT_SIZE};
    initializeParams.pPmSamplingObject = m_pmSamplerObject;
    initializeParams.counterDataSize = counterDataImage.size();
    initializeParams.pCounterData = counterDataImage.data();
    CUPTI_API_CALL(
        cuptiPmSamplingCounterDataImageInitialize(&initializeParams));
    return CUPTI_SUCCESS;
  }

  /**
   * @brief Reset a counter data image for reuse.
   * @param counterDataImage Buffer to reset.
   * @return CUPTI result status.
   */
  CUptiResult ResetCounterDataImage(std::vector<uint8_t> &counterDataImage) {
    CUpti_PmSampling_CounterDataImage_Initialize_Params initializeParams{
        CUpti_PmSampling_CounterDataImage_Initialize_Params_STRUCT_SIZE};
    initializeParams.pPmSamplingObject = m_pmSamplerObject;
    initializeParams.counterDataSize = counterDataImage.size();
    initializeParams.pCounterData = counterDataImage.data();
    CUPTI_API_CALL(
        cuptiPmSamplingCounterDataImageInitialize(&initializeParams));
    return CUPTI_SUCCESS;
  }

  /**
   * @brief Enable PM sampling on a device.
   * @param devIndex Index of the CUDA device.
   * @return CUPTI result status.
   */
  CUptiResult EnablePmSampling(size_t devIndex) {
    CUpti_PmSampling_Enable_Params enableParams{
        CUpti_PmSampling_Enable_Params_STRUCT_SIZE};
    enableParams.deviceIndex = devIndex;
    CUPTI_API_CALL(cuptiPmSamplingEnable(&enableParams));
    m_pmSamplerObject = enableParams.pPmSamplingObject;
    return CUPTI_SUCCESS;
  }

  /**
   * @brief Disable PM sampling.
   * @return CUPTI result status.
   */
  CUptiResult DisablePmSampling() {
    CUpti_PmSampling_Disable_Params disableParams{
        CUpti_PmSampling_Disable_Params_STRUCT_SIZE};
    disableParams.pPmSamplingObject = m_pmSamplerObject;
    CUPTI_API_CALL(cuptiPmSamplingDisable(&disableParams));
    return CUPTI_SUCCESS;
  }

  /**
   * @brief Configure PM sampling parameters.
   * @param configImage Configuration data.
   * @param hardwareBufferSize Size of hardware buffer in bytes.
   * @param samplingInterval Sampling interval in GPU sysclk cycles.
   * @return CUPTI result status.
   */
  CUptiResult SetConfig(std::vector<uint8_t> &configImage,
                        size_t hardwareBufferSize, uint64_t samplingInterval) {
    CUpti_PmSampling_SetConfig_Params setConfigParams = {
        CUpti_PmSampling_SetConfig_Params_STRUCT_SIZE};
    setConfigParams.pPmSamplingObject = m_pmSamplerObject;

    setConfigParams.configSize = configImage.size();
    setConfigParams.pConfig = configImage.data();

    setConfigParams.hardwareBufferSize = hardwareBufferSize;
    setConfigParams.samplingInterval = samplingInterval;

    setConfigParams.triggerMode = CUpti_PmSampling_TriggerMode::
        CUPTI_PM_SAMPLING_TRIGGER_MODE_GPU_SYSCLK_INTERVAL;
    CUPTI_API_CALL(cuptiPmSamplingSetConfig(&setConfigParams));
    return CUPTI_SUCCESS;
  }

  /**
   * @brief Start PM sampling.
   * @return CUPTI result status.
   */
  CUptiResult StartPmSampling() {
    CUpti_PmSampling_Start_Params startProfilingParams = {
        CUpti_PmSampling_Start_Params_STRUCT_SIZE};
    startProfilingParams.pPmSamplingObject = m_pmSamplerObject;
    CUPTI_API_CALL(cuptiPmSamplingStart(&startProfilingParams));
    return CUPTI_SUCCESS;
  }

  /**
   * @brief Stop PM sampling.
   * @return CUPTI result status.
   */
  CUptiResult StopPmSampling() {
    CUpti_PmSampling_Stop_Params stopProfilingParams = {
        CUpti_PmSampling_Stop_Params_STRUCT_SIZE};
    stopProfilingParams.pPmSamplingObject = m_pmSamplerObject;
    CUPTI_API_CALL(cuptiPmSamplingStop(&stopProfilingParams));
    return CUPTI_SUCCESS;
  }

  /**
   * @brief Decode raw PM sampling data.
   * @param counterDataImage Buffer containing raw counter data.
   * @return CUPTI result status.
   */
  CUptiResult DecodePmSamplingData(std::vector<uint8_t> &counterDataImage) {
    CUpti_PmSampling_DecodeData_Params decodeDataParams = {
        CUpti_PmSampling_DecodeData_Params_STRUCT_SIZE};
    decodeDataParams.pPmSamplingObject = m_pmSamplerObject;
    decodeDataParams.pCounterDataImage = counterDataImage.data();
    decodeDataParams.counterDataImageSize = counterDataImage.size();
    CUPTI_API_CALL(cuptiPmSamplingDecodeData(&decodeDataParams));
    return CUPTI_SUCCESS;
  }

  /**
   * @brief Get the underlying PM sampler object.
   * @return Pointer to the PM sampler object.
   */
  CUpti_PmSampling_Object *GetPmSamplerObject() { return m_pmSamplerObject; }

  /**
   * @brief Get the chip name for a CUDA device.
   * @param deviceIndex Index of the CUDA device.
   * @param chipName Output string for chip name.
   * @return CUPTI result status.
   */
  static CUptiResult GetChipName(size_t deviceIndex, std::string &chipName) {
    CUpti_Profiler_Initialize_Params profilerInitializeParams = {
        CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
    CUPTI_API_CALL(cuptiProfilerInitialize(&profilerInitializeParams));

    CUpti_Device_GetChipName_Params getChipNameParams = {
        CUpti_Device_GetChipName_Params_STRUCT_SIZE};
    getChipNameParams.deviceIndex = deviceIndex;
    CUPTI_API_CALL(cuptiDeviceGetChipName(&getChipNameParams));
    chipName = getChipNameParams.pChipName;
    return CUPTI_SUCCESS;
  }

  /**
   * @brief Get counter availability image for a device.
   * @param deviceIndex Index of the CUDA device.
   * @param counterAvailibilityImage Output buffer for availability data.
   * @return CUPTI result status.
   */
  static CUptiResult
  GetCounterAvailabilityImage(size_t deviceIndex,
                              std::vector<uint8_t> &counterAvailibilityImage) {
    CUpti_PmSampling_GetCounterAvailability_Params getCounterAvailabilityParams{
        CUpti_PmSampling_GetCounterAvailability_Params_STRUCT_SIZE};
    getCounterAvailabilityParams.deviceIndex = deviceIndex;
    CUPTI_API_CALL(
        cuptiPmSamplingGetCounterAvailability(&getCounterAvailabilityParams));

    counterAvailibilityImage.clear();
    counterAvailibilityImage.resize(
        getCounterAvailabilityParams.counterAvailabilityImageSize);
    getCounterAvailabilityParams.pCounterAvailabilityImage =
        counterAvailibilityImage.data();
    CUPTI_API_CALL(
        cuptiPmSamplingGetCounterAvailability(&getCounterAvailabilityParams));
    return CUPTI_SUCCESS;
  }

private:
  CUptiResult InitializeProfiler() {
    CUpti_Profiler_Initialize_Params profilerInitializeParams = {
        CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
    CUPTI_API_CALL(cuptiProfilerInitialize(&profilerInitializeParams));
    return CUPTI_SUCCESS;
  }

  CUptiResult DeInitializeProfiler() {
    CUpti_Profiler_DeInitialize_Params profilerDeInitializeParams = {
        CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE};
    CUPTI_API_CALL(cuptiProfilerDeInitialize(&profilerDeInitializeParams));
    return CUPTI_SUCCESS;
  }
};
