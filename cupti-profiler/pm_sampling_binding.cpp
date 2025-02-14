#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pm_sampling.h"
#include <cuda.h>

// We may need the following if you plan to handle exceptions or convert them to
// Python exceptions #include <pybind11/exception_translator.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

/**
 * @brief Python bindings for CUPTI PM Sampling functionality
 *
 * This class provides a Python-friendly interface to the CUPTI PM sampling
 * features, allowing performance metric sampling of CUDA workloads from Python
 * code.
 */
class PmSampler {
public:
  /**
   * @brief Constructor
   *
   * Initializes CUDA and CUPTI for the given device index, discovers chip name
   * and counter availability, and sets up the profiler host.
   *
   * @param deviceIndex CUDA device index to profile
   * @throws std::runtime_error if initialization fails
   */
  PmSampler(int deviceIndex) : m_deviceIndex(deviceIndex) {
    // Initialize CUDA driver
    DRIVER_API_CALL(cuInit(0));
    DRIVER_API_CALL(cuDeviceGet(&m_cuDevice, deviceIndex));

    // Check device support
    {
      CUpti_Profiler_DeviceSupported_Params params = {
          CUpti_Profiler_DeviceSupported_Params_STRUCT_SIZE};
      params.cuDevice = m_cuDevice;
      params.api = CUPTI_PROFILER_PM_SAMPLING;
      CUPTI_API_CALL(cuptiProfilerDeviceSupported(&params));
      if (params.isSupported != CUPTI_PROFILER_CONFIGURATION_SUPPORTED) {
        throw std::runtime_error(
            "PM Sampling not supported on this device configuration.");
      }
    }

    // Get chip name and counter availability
    std::string chipName;
    CuptiPmSampling::GetChipName(deviceIndex, chipName);
    CuptiPmSampling::GetCounterAvailabilityImage(deviceIndex,
                                                 m_counterAvailability);

    // Set up the profiler host
    m_profilerHost.SetUp(chipName, m_counterAvailability);
  }

  /**
   * @brief Destructor
   *
   * Ensures sampling is disabled and resources are cleaned up.
   */
  ~PmSampler() {
    if (m_samplingEnabled) {
      disable_sampling();
    }
    m_profilerHost.TearDown();
  }

  /**
   * @brief Query available base metrics
   *
   * @return List of available base metric names
   */
  std::vector<std::string> query_base_metrics() {
    std::vector<std::string> metrics;
    m_profilerHost.GetSupportedBaseMetrics(metrics);
    return metrics;
  }

  /**
   * @brief Query properties of specific metrics
   *
   * @param metricNames List of metric names to query
   * @return Dictionary mapping metric names to their properties
   */
  std::unordered_map<std::string, std::unordered_map<std::string, std::string>>
  query_metric_properties(const std::vector<std::string> &metricNames) {
    std::unordered_map<std::string,
                       std::unordered_map<std::string, std::string>>
        result;

    for (const auto &metric : metricNames) {
      CUpti_MetricType metricType;
      std::string description;
      m_profilerHost.GetMetricProperties(metric, metricType, description);

      std::string metricTypeStr;
      switch (metricType) {
      case CUPTI_METRIC_TYPE_COUNTER:
        metricTypeStr = "Counter";
        break;
      case CUPTI_METRIC_TYPE_THROUGHPUT:
        metricTypeStr = "Throughput";
        break;
      case CUPTI_METRIC_TYPE_RATIO:
        metricTypeStr = "Ratio";
        break;
      default:
        metricTypeStr = "Unknown";
        break;
      }

      std::unordered_map<std::string, std::string> props;
      props["description"] = description;
      props["type"] = metricTypeStr;
      result[metric] = props;
    }

    return result;
  }

  /**
   * @brief Enable PM sampling for specified metrics
   *
   * @param metrics List of metrics to sample
   * @param samplingInterval Sampling interval in GPU sysclk cycles
   * @param hardwareBufferSize Size of hardware buffer in bytes
   * @param maxSamples Maximum number of samples to store
   */
  void enable_sampling(const std::vector<std::string> &metrics,
                       uint64_t samplingInterval = 100000,
                       size_t hardwareBufferSize = 512ULL * 1024ULL * 1024ULL,
                       uint64_t maxSamples = 10000) {
    if (m_samplingEnabled)
      throw std::runtime_error("Sampling is already enabled.");

    // Convert std::vector<std::string> to std::vector<const char*>
    m_metrics = metrics;
    m_metricPointers.clear();
    m_metricPointers.reserve(metrics.size());
    for (auto &m : m_metrics)
      m_metricPointers.push_back(m.c_str());

    // Create config image
    m_profilerHost.CreateConfigImage(m_metricPointers, m_configImage);

    // Initialize PM sampling
    m_pmSampler.SetUp(m_deviceIndex);
    m_pmSampler.EnablePmSampling(m_deviceIndex);
    m_pmSampler.SetConfig(m_configImage, hardwareBufferSize, samplingInterval);

    // Create counter data image
    m_pmSampler.CreateCounterDataImage(maxSamples, m_metricPointers,
                                       m_counterDataImage);

    m_samplingEnabled = true;
  }

  /**
   * @brief Disable PM sampling and clean up resources
   */
  void disable_sampling() {
    if (!m_samplingEnabled)
      return;
    m_pmSampler.DisablePmSampling();
    m_pmSampler.TearDown();
    m_samplingEnabled = false;
  }

  /**
   * @brief Start collecting samples
   * @throws std::runtime_error if sampling is not enabled
   */
  void start_sampling() {
    if (!m_samplingEnabled)
      throw std::runtime_error(
          "Sampling has not been enabled. Call enable_sampling() first.");
    m_pmSampler.StartPmSampling();
  }

  /**
   * @brief Stop collecting samples
   * @throws std::runtime_error if sampling is not enabled
   */
  void stop_sampling() {
    if (!m_samplingEnabled)
      throw std::runtime_error(
          "Sampling has not been enabled. Call enable_sampling() first.");
    m_pmSampler.StopPmSampling();
  }

  /**
   * @brief Get collected samples
   *
   * @return List of samples, each containing timestamps and metric values
   * @throws std::runtime_error if sampling is not enabled
   */
  std::vector<py::dict> get_samples() {
    if (!m_samplingEnabled)
      throw std::runtime_error(
          "Sampling has not been enabled. Call enable_sampling() first.");

    // Decode the data
    CUPTI_API_CALL(m_pmSampler.DecodePmSamplingData(m_counterDataImage));

    // Get info on completed samples
    CUpti_PmSampling_GetCounterDataInfo_Params counterDataInfo{
        CUpti_PmSampling_GetCounterDataInfo_Params_STRUCT_SIZE};
    counterDataInfo.pCounterDataImage = m_counterDataImage.data();
    counterDataInfo.counterDataImageSize = m_counterDataImage.size();
    CUptiResult status = cuptiPmSamplingGetCounterDataInfo(&counterDataInfo);
    if (status != CUPTI_SUCCESS) {
      const char *errstr;
      cuptiGetResultString(status, &errstr);
      throw std::runtime_error(
          std::string("cuptiPmSamplingGetCounterDataInfo failed: ") + errstr);
    }

    // Process samples
    std::vector<py::dict> samples;
    samples.reserve(counterDataInfo.numCompletedSamples);

    for (size_t sampleIndex = 0;
         sampleIndex < counterDataInfo.numCompletedSamples; ++sampleIndex) {
      // Get sample timestamps
      CUpti_PmSampling_CounterData_GetSampleInfo_Params getSampleInfoParams = {
          CUpti_PmSampling_CounterData_GetSampleInfo_Params_STRUCT_SIZE};
      getSampleInfoParams.pPmSamplingObject = m_pmSampler.GetPmSamplerObject();
      getSampleInfoParams.pCounterDataImage = m_counterDataImage.data();
      getSampleInfoParams.counterDataImageSize = m_counterDataImage.size();
      getSampleInfoParams.sampleIndex = sampleIndex;
      CUPTI_API_CALL(
          cuptiPmSamplingCounterDataGetSampleInfo(&getSampleInfoParams));

      // Evaluate metric values
      std::vector<double> metricValues(m_metricPointers.size());
      CUpti_Profiler_Host_EvaluateToGpuValues_Params evaluateParams{
          CUpti_Profiler_Host_EvaluateToGpuValues_Params_STRUCT_SIZE};
      evaluateParams.pHostObject = m_profilerHostObject();
      evaluateParams.pCounterDataImage = m_counterDataImage.data();
      evaluateParams.counterDataImageSize = m_counterDataImage.size();
      evaluateParams.ppMetricNames = m_metricPointers.data();
      evaluateParams.numMetrics = m_metricPointers.size();
      evaluateParams.rangeIndex = sampleIndex;
      evaluateParams.pMetricValues = metricValues.data();

      CUPTI_API_CALL(cuptiProfilerHostEvaluateToGpuValues(&evaluateParams));

      // Build Python dict
      py::dict sample;
      sample["startTimestamp"] = getSampleInfoParams.startTimestamp;
      sample["endTimestamp"] = getSampleInfoParams.endTimestamp;

      py::dict metricDict;
      for (size_t i = 0; i < m_metricPointers.size(); i++) {
        metricDict[m_metricPointers[i]] = metricValues[i];
      }
      sample["metrics"] = metricDict;
      samples.push_back(sample);
    }

    // Reset for next batch
    m_pmSampler.ResetCounterDataImage(m_counterDataImage);

    return samples;
  }

private:
  // Helper to get the profiler host object
  CUpti_Profiler_Host_Object *m_profilerHostObject() {
    return m_profilerHost.GetHostObject();
  }

private:
  int m_deviceIndex = 0;
  CUdevice m_cuDevice = 0;
  bool m_samplingEnabled = false;

  CuptiProfilerHost m_profilerHost;
  CuptiPmSampling m_pmSampler;

  // Storage for configuration and data
  std::vector<uint8_t> m_counterAvailability;
  std::vector<uint8_t> m_configImage;
  std::vector<uint8_t> m_counterDataImage;

  // Storage for metric names and pointers
  std::vector<std::string> m_metrics;
  std::vector<const char *> m_metricPointers;
};

// -------------------------------------------
// PYBIND11 MODULE DEFINITION
// -------------------------------------------
PYBIND11_MODULE(pm_sampling, m) {
  m.doc() = "Python bindings for NVIDIA CUPTI PM Sampling";

  py::class_<PmSampler>(m, "PmSampler")
      .def(py::init<int>(), py::arg("device_index") = 0,
           R"pbdoc(
                Initialize a PmSampler for the given CUDA device index.
                This sets up the host side profiling objects for querying metrics and
                preparing PM sampling.
             )pbdoc")
      .def("query_base_metrics", &PmSampler::query_base_metrics,
           R"pbdoc(
                Return a list of all supported base metrics on this device.
             )pbdoc")
      .def("query_metric_properties", &PmSampler::query_metric_properties,
           py::arg("metric_names"),
           R"pbdoc(
                For each metric in the given list, return a dict describing its type and description.
                Return format:
                  {
                    metric_name: {
                      "description": <string>,
                      "type": <string>
                    },
                    ...
                  }
             )pbdoc")
      .def("enable_sampling", &PmSampler::enable_sampling, py::arg("metrics"),
           py::arg("sampling_interval") = 100000,
           py::arg("hardware_buffer_size") = 512ULL * 1024ULL * 1024ULL,
           py::arg("max_samples") = 10000,
           R"pbdoc(
                Enable PM sampling for a given list of metric names.
                sampling_interval: sampling interval in GPU sysclk cycles
                hardware_buffer_size: size in bytes of the internal hardware buffer
                max_samples: maximum number of samples to store in the host counter data image
             )pbdoc")
      .def("disable_sampling", &PmSampler::disable_sampling,
           R"pbdoc(
                Disable PM sampling and free the underlying resources.
             )pbdoc")
      .def("start_sampling", &PmSampler::start_sampling,
           R"pbdoc(
                Start PM sampling. PM sampling must have been enabled first.
             )pbdoc")
      .def("stop_sampling", &PmSampler::stop_sampling,
           R"pbdoc(
                Stop PM sampling.
             )pbdoc")
      .def("get_samples", &PmSampler::get_samples,
           R"pbdoc(
                Decode the PM sampling data, returning a list of samples.
                Each sample is a dict with fields:
                  {
                    "startTimestamp": <int>,
                    "endTimestamp":   <int>,
                    "metrics": {
                       "metricName": <double>,
                       ...
                    }
                  }
                The counter data image is reset after this call, so subsequent
                calls will fetch fresh data.
             )pbdoc");
}
