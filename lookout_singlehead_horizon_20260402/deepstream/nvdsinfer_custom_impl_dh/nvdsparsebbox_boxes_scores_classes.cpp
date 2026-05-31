/*
Custom DeepStream bbox parser for models that output separate blobs:
  - boxes:   [B,N,4] or [N,4] or [N,1,4]   (cx,cy,w,h) in network input pixels
  - scores:  [B,N,1] or [N,1] or [N]       (confidence)
  - classes: [B,N,1] or [N,1]              (class id as float)

This matches the ONNX exported by:
  - tools/export_ds_multioutput_onnx.py (boxes/scores/classes + extra heads)
  - YOLOv7-DL23/export_yoloV7_pack_dh.py (boxes/packed_scores/classes)

Extra outputs (dist/heading/...) are ignored by this parser but can be accessed through
NvDsInferTensorMeta when `output-tensor-meta=1` is enabled.
*/

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "nvdsinfer_custom_impl.h"

static inline float clampf(float v, float lo, float hi) { return std::min(hi, std::max(lo, v)); }

static const NvDsInferLayerInfo* findLayer(const std::vector<NvDsInferLayerInfo>& layers, const char* name) {
    for (auto const& l : layers) {
        if (l.layerName && std::strcmp(l.layerName, name) == 0) return &l;
    }
    return nullptr;
}

static inline int64_t numElements(const NvDsInferLayerInfo& l) {
    return static_cast<int64_t>(l.inferDims.numElements);
}

extern "C" bool NvDsInferParseCustomBoxesScoresClasses(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    objectList.clear();

    const NvDsInferLayerInfo* boxesL = findLayer(outputLayersInfo, "boxes");
    const NvDsInferLayerInfo* scoresL = findLayer(outputLayersInfo, "scores");
    const NvDsInferLayerInfo* classesL = findLayer(outputLayersInfo, "classes");

    // Fallback to positional if names are not present.
    if (!boxesL && outputLayersInfo.size() >= 1) boxesL = &outputLayersInfo[0];
    if (!scoresL && outputLayersInfo.size() >= 2) scoresL = &outputLayersInfo[1];
    if (!classesL && outputLayersInfo.size() >= 3) classesL = &outputLayersInfo[2];

    if (!boxesL || !scoresL || !classesL) {
        std::cerr << "NvDsInferParseCustomBoxesScoresClasses: missing required output layers\n";
        return false;
    }

    const float* boxes = static_cast<const float*>(boxesL->buffer);
    const float* scores = static_cast<const float*>(scoresL->buffer);
    const float* classes = static_cast<const float*>(classesL->buffer);

    const int64_t nb = numElements(*boxesL);
    const int64_t ns = numElements(*scoresL);
    const int64_t nc = numElements(*classesL);

    if (nb <= 0 || ns <= 0 || nc <= 0) return true;
    if (nb % 4 != 0) {
        std::cerr << "NvDsInferParseCustomBoxesScoresClasses: boxes numElements not divisible by 4\n";
        return false;
    }

    const int64_t N = nb / 4;
    if (ns < N || nc < N) {
        std::cerr << "NvDsInferParseCustomBoxesScoresClasses: scores/classes smaller than boxes\n";
        return false;
    }

    const float netW = static_cast<float>(networkInfo.width);
    const float netH = static_cast<float>(networkInfo.height);
    const int numClasses = static_cast<int>(detectionParams.numClassesConfigured);

    for (int64_t i = 0; i < N; ++i) {
        const float cx = boxes[i * 4 + 0];
        const float cy = boxes[i * 4 + 1];
        const float w = boxes[i * 4 + 2];
        const float h = boxes[i * 4 + 3];

        float left = cx - w / 2.0f;
        float top = cy - h / 2.0f;
        float width = w;
        float height = h;

        left = clampf(left, 0.0f, netW);
        top = clampf(top, 0.0f, netH);
        width = clampf(width, 0.0f, netW - left);
        height = clampf(height, 0.0f, netH - top);

        if (width < 1.0f || height < 1.0f) continue;

        const float conf = scores[i];
        int cls = static_cast<int>(std::lround(classes[i]));
        if (cls < 0) cls = 0;
        if (numClasses > 0 && cls >= numClasses) cls = numClasses - 1;

        float thres = 0.0f;
        if (numClasses > 0 && detectionParams.perClassPreclusterThreshold.size() > static_cast<size_t>(cls)) {
            thres = detectionParams.perClassPreclusterThreshold[static_cast<size_t>(cls)];
        }
        if (conf < thres) continue;

        NvDsInferParseObjectInfo obj;
        obj.left = left;
        obj.top = top;
        obj.width = width;
        obj.height = height;
        obj.detectionConfidence = conf;
        obj.classId = cls;
        objectList.push_back(obj);
    }

    return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomBoxesScoresClasses);
