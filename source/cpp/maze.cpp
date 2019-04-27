
/* maze.cpp */

#include <iostream>
#include <vector>

#include "maze.hpp"
#include "agent.hpp"

int main(int argc, char** argv)
{
    const std::vector<MazeCellType> mazeGrid {
        Normal, Normal, Normal, Reward,
        Normal, Block,  Normal, Damage,
        Normal, Normal, Normal, Normal,
    };

    MazeEnvironment mazeEnv { 3, 4, mazeGrid, 0.8f };
    RandomAgent<MazeState, MazeAction> agent { mazeEnv.AvailableActions() };

    for (int i = 0; i < 10; ++i) {
        MazeState state = mazeEnv.Reset();
        float totalReward = 0.0f;
        bool done = false;

        while (!done) {
            MazeAction action = agent.Policy(state);

            std::optional<MazeState> nextState;
            std::optional<float> reward;
            std::tie(nextState, reward, done) = mazeEnv.Step(action);

            if (reward)
                totalReward += reward.value();
            
            if (nextState)
                state = nextState.value();
        }

        std::cout << "Episode " << i << ": Agent gets " << totalReward << " reward\n";
    }

    return 0;
}
