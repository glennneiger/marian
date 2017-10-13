#pragma once

#include "matrix_wrapper.h"

namespace amunmt {
namespace GPU {

struct NthOut
{
  uint ind;
  float score;

  __device__ __host__
  NthOut() {}

  __device__ __host__
  NthOut(uint vInd, float vScore)
  :ind(vInd)
  ,score(vScore)
  {}

  __device__ __host__
  NthOut& operator+=(const NthOut& rhs)
  {
    ind += rhs.ind;
    score += rhs.score;
    return *this;
  }

  __device__ __host__
  float& operator=(const float& rhs)
  {
    // only to be used to init variable in matrix.h gSum
    assert(rhs == 0.0f);
    score = rhs;
    return score;
  }

};

/////////////////////////////////////////////////////////////////////////////////////////

struct NthOutBatch : public NthOut
{
  uint batch;
  uint vocabId;

  __device__ __host__
  NthOutBatch() {}

  __device__ __host__
  NthOutBatch(uint vInd, float vScore, uint vBatch, uint vVocabId)
  :NthOut(vInd, vScore)
  ,batch(vBatch)
  ,vocabId(vVocabId)
  {}

  __device__ __host__
  NthOutBatch& operator+=(const NthOutBatch& rhs)
  {
    ind += rhs.ind;
    score += rhs.score;
    batch += rhs.batch;
    vocabId += rhs.vocabId;
    return *this;
  }

  __device__ __host__
  float& operator=(const float& rhs)
  {
    // only to be used to init variable in matrix.h gSum
    assert(rhs == 0.0f);
    score = rhs;
    return score;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////

inline std::ostream& operator<<(std::ostream &out, const NthOut &obj)
{
  out << "(" << obj.ind << "," << obj.score << ")";
  return out;
}

inline std::ostream& operator<<(std::ostream &out, const NthOutBatch &obj)
{
  out << "("
      << obj.ind << ","
      << obj.score << ","
      << obj.batch << ","
      << obj.vocabId << ")";
  return out;
}

/////////////////////////////////////////////////////////////////////////////////////////

__global__ void gMaxElement(mblas::MatrixWrapper<NthOut> out,
                            const mblas::MatrixWrapper<float> probsWrap,
                            const mblas::MatrixWrapper<uint> batchPositionWrap,
                            uint numBatches);

__global__ void gMaxElementUpdate(mblas::MatrixWrapper<NthOut> out,
                                  mblas::MatrixWrapper<float> probsWrap,
                                  mblas::MatrixWrapper<NthOut> resNewWrap,
                                  const mblas::MatrixWrapper<uint> batchPositionWrap,
                                  const mblas::MatrixWrapper<uint> cumBeamSizesWrap,
                                  uint numBlocks);

__global__ void gGetValueByKey(mblas::MatrixWrapper<float> out,
                              const   mblas::MatrixWrapper<float> in,
                              uint* indices, uint n);

}
}
