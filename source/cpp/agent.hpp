
/* agent.hpp */

#ifndef AGENT_HPP
#define AGENT_HPP

#include <vector>

/*
 * RandomAgentクラス
 */

template <typename TState, typename TAction>
class RandomAgent
{
public:
    RandomAgent(const std::vector<TAction>& availableActions);

    TAction Policy(TState state);

private:
    std::vector<TAction> mActions;
};

template <typename TState, typename TAction>
RandomAgent<TState, TAction>::RandomAgent(const std::vector<TAction>& availableActions) :
    mActions(availableActions)
{
}

template <typename TState, typename TAction>
TAction RandomAgent<TState, TAction>::Policy(TState state)
{
    return this->mActions[RandomChoice(0, this->mActions.size() - 1)];
}

#endif /* AGENT_HPP */
