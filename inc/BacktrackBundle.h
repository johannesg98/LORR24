#pragma once
#include "common.h"
#include <unordered_set>

struct BacktrackBundle
{
    int start_time;
    std::unordered_set<int> agents_unfinished;
    std::vector<int> traveltimes;

    BacktrackBundle(int start_time)
        : start_time(start_time){}
};
