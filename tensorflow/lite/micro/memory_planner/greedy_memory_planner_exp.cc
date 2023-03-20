#include "tensorflow/lite/micro/memory_planner/greedy_memory_planner.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace{
    constexpr int kScratchBufferSize = 4096;
    alignas(4) unsigned char g_scratch_buffer[kScratchBufferSize];
}

TF_LITE_MICRO_TESTS_BEGIN
TF_LITE_MICRO_TEST(origintest){
    tflite::GreedyMemoryPlanner planner;
    planner.Init(g_scratch_buffer,kScratchBufferSize);
    planner.AddBuffer(12*1024,0,0);
    planner.AddBuffer(96*1024,0,4);
    planner.AddBuffer(32*1024,1,2);
    planner.AddBuffer(64*1024,2,3);
    planner.AddBuffer(16*1024,3,7);
    planner.AddBuffer(32*1024,4,5);
    planner.AddBuffer(64*1024,5,6);
    planner.AddBuffer(16*1024,6,7);
    planner.AddBuffer(32*1024,7,8);
    planner.PrintMemoryPlan();
    MicroPrintf("pm is %d",planner.GetMaximumMemorySize());
}

TF_LITE_MICRO_TEST(Optimaltest){
    tflite::GreedyMemoryPlanner planner;
    planner.Init(g_scratch_buffer,kScratchBufferSize);
    planner.AddBuffer(12*1024,0,0);
    planner.AddBuffer(96*1024,0,2);
    planner.AddBuffer(32*1024,1,3);
    planner.AddBuffer(64*1024,3,4);
    planner.AddBuffer(16*1024,4,7);
    planner.AddBuffer(32*1024,2,5);
    planner.AddBuffer(64*1024,5,6);
    planner.AddBuffer(16*1024,6,7);
    planner.AddBuffer(32*1024,7,8);
    MicroPrintf("pm is %d",planner.GetMaximumMemorySize());
}

TF_LITE_MICRO_TEST(REtest){
    tflite::GreedyMemoryPlanner planner;
    planner.Init(g_scratch_buffer,kScratchBufferSize);
    planner.AddBuffer(12*1024,0,4);
    planner.AddBuffer(96*1024,0,1);
    planner.AddBuffer(96*1024,4,5);
    planner.AddBuffer(32*1024,1,2);
    planner.AddBuffer(64*1024,2,3);
    planner.AddBuffer(16*1024,3,8);
    planner.AddBuffer(32*1024,5,6);
    planner.AddBuffer(64*1024,6,7);
    planner.AddBuffer(16*1024,7,8);
    planner.AddBuffer(32*1024,8,9);
    MicroPrintf("pm is %d",planner.GetMaximumMemorySize());
    
}

TF_LITE_MICRO_TESTS_END