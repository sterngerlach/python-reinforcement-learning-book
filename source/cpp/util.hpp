
/* util.hpp */

#ifndef UTIL_HPP
#define UTIL_HPP

#include <random>
#include <vector>

/*
 * 乱数
 */

int RandomChoice(int minValue, int maxValue)
{
    static std::default_random_engine randomEngine;
    std::uniform_int_distribution<int> uniformDist { minValue, maxValue - 1 };
    return uniformDist(randomEngine);
}

int RandomChoice(const std::vector<float>& prob)
{
    static std::default_random_engine randomEngine;
    std::discrete_distribution<int> discreteDist { prob.cbegin(), prob.cend() };
    return discreteDist(randomEngine);
}

#endif /* UTIL_HPP */
