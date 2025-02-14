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

class PmSampler {
public:
  /**
   * Constructor:
   *   1) Initializes CUDA and CUPTI for the given device index.
   *   2) Discovers the chip name and counter availability image.
   *   3) Sets up the CuptiProfilerHost so that we can later create config
   * images, query metrics, etc.
   */
  PmSampler(int deviceIndex) : m_deviceIndex(deviceIndex) {
    // Initialize the CUDA driver
    DRIVER_API_CALL(cuInit(0));

    // Retrieve the device handle
    DRIVER_API_CALL(cuDeviceGet(&m_cuDevice, deviceIndex));

    // Some basic device checks (optional). For example, check device support:
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

    // Grab the chip name
    std::string chipName;
    CuptiPmSampling::GetChipName(deviceIndex, chipName);

    // Grab the counter availability image
    CuptiPmSampling::GetCounterAvailabilityImage(deviceIndex,
                                                 m_counterAvailability);

    // Set up the CuptiProfilerHost for this chip and availability image
    m_profilerHost.SetUp(chipName, m_counterAvailability);
  }

  /**
   * Destructor:
   *   1) Ensure we disable sampling (if still active).
   *   2) Teardown the CuptiPmSampling and CuptiProfilerHost.
   */
  ~PmSampler() {
    // If sampling is enabled, disable it
    if (m_samplingEnabled) {
      disable_sampling();
    }
    // Tear down the profiler host if it was set up
    m_profilerHost.TearDown();
  }

  /**
   * query_base_metrics():
   *   - Returns a list of all base metrics reported by the CUPTI profiler for
   * this device.
   */
  std::vector<std::string> query_base_metrics() {
    std::vector<std::string> metrics;
    m_profilerHost.GetSupportedBaseMetrics(metrics);
    return metrics;
  }

  /**
   * query_metric_properties():
   *   - For each metric name, return a dictionary of { "description": ...,
   * "type": ... }.
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

      // Convert metric type to string
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
   * enable_sampling(metrics, sampling_interval, hardware_buffer_size,
   * max_samples): 1) Create the config image for the given set of metrics. 2)
   * Initialize and enable PM sampling for the device. 3) Set the config
   * (interval, buffer size, etc.). 4) Create the counter data image that will
   * store the samples.
   */
  void enable_sampling(const std::vector<std::string> &metrics,
                       uint64_t samplingInterval = 100000, // 100us
                       size_t hardwareBufferSize = 512ULL * 1024ULL * 1024ULL,
                       uint64_t maxSamples = 10000) {
    if (m_samplingEnabled)
      throw std::runtime_error("Sampling is already enabled.");

    // Convert std::vector<std::string> to std::vector<const char*>
    // We need to keep these strings in memory, so store them in a member
    m_metrics = metrics;
    m_metricPointers.clear();
    m_metricPointers.reserve(metrics.size());
    for (auto &m : m_metrics)
      m_metricPointers.push_back(m.c_str());

    // 1. Create config image
    m_profilerHost.CreateConfigImage(m_metricPointers, m_configImage);

    // 2. Initialize the CuptiPmSampling object
    m_pmSampler.SetUp(m_deviceIndex);

    // 3. Enable PM sampling
    m_pmSampler.EnablePmSampling(m_deviceIndex);

    // 4. Set config (interval, buffer, etc.)
    m_pmSampler.SetConfig(m_configImage, hardwareBufferSize, samplingInterval);

    // 5. Create counter data image
    m_pmSampler.CreateCounterDataImage(maxSamples, m_metricPointers,
                                       m_counterDataImage);

    m_samplingEnabled = true;
  }

  /**
   * disable_sampling():
   *   - Disable PM sampling and tear down the CuptiPmSampling object.
   */
  void disable_sampling() {
    if (!m_samplingEnabled)
      return;
    m_pmSampler.DisablePmSampling();
    m_pmSampler.TearDown();
    m_samplingEnabled = false;
  }

  /**
   * start_sampling():
   *   - Simply call start on the PM sampling.
   */
  void start_sampling() {
    if (!m_samplingEnabled)
      throw std::runtime_error(
          "Sampling has not been enabled. Call enable_sampling() first.");
    m_pmSampler.StartPmSampling();
  }

  /**
   * stop_sampling():
   *   - Stop the PM sampling (so that no new samples come in).
   */
  void stop_sampling() {
    if (!m_samplingEnabled)
      throw std::runtime_error(
          "Sampling has not been enabled. Call enable_sampling() first.");
    m_pmSampler.StopPmSampling();
  }

  /**
   * get_samples():
   *   - Decode the data in the counterDataImage, retrieving all completed
   * samples.
   *   - For each completed sample, EvaluateCounterData and build a
   * Python-friendly structure:
   *       {
   *           "startTimestamp": <uint64_t>,
   *           "endTimestamp":   <uint64_t>,
   *           "metrics":        { "metric_name": <value>, ... }
   *       }
   *   - Then reset (so that future calls to get_samples() will fetch newly
   * accumulated samples).
   *   - Return a list of samples.
   *
   * Note: For a "continuous" scenario you'd typically call decode repeatedly
   *       (perhaps on another thread) to not overflow the internal hardware
   * buffer.
   */
  std::vector<py::dict> get_samples() {
    if (!m_samplingEnabled)
      throw std::runtime_error(
          "Sampling has not been enabled. Call enable_sampling() first.");

    // 1. Decode the data
    CUPTI_API_CALL(m_pmSampler.DecodePmSamplingData(m_counterDataImage));

    // 2. Get info on how many samples are completed
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

    // 3. Evaluate the samples, gather them in a list
    std::vector<py::dict> samples;
    samples.reserve(counterDataInfo.numCompletedSamples);

    for (size_t sampleIndex = 0;
         sampleIndex < counterDataInfo.numCompletedSamples; ++sampleIndex) {
      // We'll reuse the EvaluateCounterData logic from CuptiProfilerHost,
      // but we also need to retrieve timestamps from
      // cuptiPmSamplingCounterDataGetSampleInfo.
      CUpti_PmSampling_CounterData_GetSampleInfo_Params getSampleInfoParams = {
          CUpti_PmSampling_CounterData_GetSampleInfo_Params_STRUCT_SIZE};
      getSampleInfoParams.pPmSamplingObject = m_pmSampler.GetPmSamplerObject();
      getSampleInfoParams.pCounterDataImage = m_counterDataImage.data();
      getSampleInfoParams.counterDataImageSize = m_counterDataImage.size();
      getSampleInfoParams.sampleIndex = sampleIndex;
      CUPTI_API_CALL(
          cuptiPmSamplingCounterDataGetSampleInfo(&getSampleInfoParams));

      // Evaluate the metric values
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

      // Build a Python dict
      py::dict sample;
      sample["startTimestamp"] = getSampleInfoParams.startTimestamp;
      sample["endTimestamp"] = getSampleInfoParams.endTimestamp;

      // Build the metrics sub-dict
      py::dict metricDict;
      for (size_t i = 0; i < m_metricPointers.size(); i++) {
        // metric name as key, value as double
        metricDict[m_metricPointers[i]] = metricValues[i];
      }
      sample["metrics"] = metricDict;
      samples.push_back(sample);
    }

    // 4. Reset the counter data image for future sampling
    m_pmSampler.ResetCounterDataImage(m_counterDataImage);

    return samples;
  }

private:
  // Helper to get the underlying CUPTI profiler host object pointer.
  // (We rely on CuptiProfilerHost having a small accessor or store the pointer
  // as needed.)
  CUpti_Profiler_Host_Object *m_profilerHostObject() {
    // We'll do a small hack: we rely on CuptiProfilerHost storing it in a
    // private field but we have a small accessor if needed. Let's add a small
    // method:
    return m_profilerHost.GetHostObject();
  }

private:
  int m_deviceIndex = 0;
  CUdevice m_cuDevice = 0;
  bool m_samplingEnabled = false;

  CuptiProfilerHost m_profilerHost;
  CuptiPmSampling m_pmSampler;

  // Storage for the config image and the sampled data
  std::vector<uint8_t> m_counterAvailability;
  std::vector<uint8_t> m_configImage;
  std::vector<uint8_t> m_counterDataImage;

  // A copy of user-provided metric names and pointers (lifetime must outlast
  // usage).
  std::vector<std::string> m_metrics;
  std::vector<const char *> m_metricPointers;
};

// -------------------------------------------
// PYBIND11 MODULE DEFINITION
// -------------------------------------------
PYBIND11_MODULE(pm_sampling, m) {
  m.doc() = "Python bindings for NVIDIA CUPTI PM Sampling (example)";

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
