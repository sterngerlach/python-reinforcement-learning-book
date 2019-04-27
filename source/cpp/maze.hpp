
/* maze.hpp */

#ifndef MAZE_HPP
#define MAZE_HPP

#include <cassert>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "util.hpp"

/*
 * MazeStateクラス
 */

class MazeState
{
public:
    MazeState(int row, int col);
    MazeState(const MazeState& other) = default;

    MazeState& operator=(const MazeState& other) = default;

    inline int Row() const { return this->mRow; }
    inline int Column() const { return this->mColumn; }

    // inline void SetRow(int row) { this->mRow = row; }
    // inline void SetColumn(int col) { this->mColumn = col; }
    
    std::string ToString() const;

    friend bool operator==(const MazeState& lhs, const MazeState& rhs);
    friend bool operator!=(const MazeState& lhs, const MazeState& rhs);
    friend bool operator<(const MazeState& lhs, const MazeState& rhs);

private:
    int mRow;
    int mColumn;
};

bool operator==(const MazeState& lhs, const MazeState& rhs);
bool operator!=(const MazeState& lhs, const MazeState& rhs);
bool operator<(const MazeState& lhs, const MazeState& rhs);

/*
 * MazeStateクラスの実装
 */

MazeState::MazeState(int row = -1, int column = -1) : 
    mRow(row),
    mColumn(column)
{
}

std::string MazeState::ToString() const
{
    std::ostringstream ostr;
    ostr << "<State: [" << this->mRow << ", " << this->mColumn << "]>";
    return ostr.str();
}

bool operator==(const MazeState& lhs, const MazeState& rhs)
{
    return (lhs.mRow == rhs.mRow) && (lhs.mColumn == rhs.mColumn);
}

bool operator!=(const MazeState& lhs, const MazeState& rhs)
{
    return !(lhs == rhs);
}

bool operator<(const MazeState& lhs, const MazeState& rhs)
{
    return (lhs.Row() < rhs.Row()) || (lhs.Column() < rhs.Column());
}

/*
 * MazeAction列挙体
 */

enum MazeAction
{
    Up = 1,
    Down = -1,
    Left = 2,
    Right = -2,
};

/*
 * MazeCellType列挙体
 */

enum MazeCellType
{
    Normal,
    Damage,
    Reward,
    Block,
};

/*
 * MazeEnvironmentクラス
 */

class MazeEnvironment
{
public:
    MazeEnvironment(int rowLength, int columnLength,
                    const std::vector<MazeCellType>& grid, float moveProb);

    inline int RowLength() const { return this->mRowLength; }
    inline int ColumnLength() const { return this->mColumnLength; }

    const std::vector<MazeAction>& AvailableActions() const;
    std::vector<MazeState> AvailableStates() const;
    bool CanActionAt(const MazeState& state) const;

    MazeCellType At(int row, int col) const;

    std::map<MazeState, float> Transition(const MazeState& state, MazeAction action);
    std::tuple<float, bool> Reward(const MazeState& state);

    MazeState Reset();

    std::tuple<std::optional<MazeState>, std::optional<float>, bool>
        Step(MazeAction action);

    std::tuple<std::optional<MazeState>, std::optional<float>, bool>
        Transit(const MazeState& state, MazeAction action);

private:
    MazeState Move(const MazeState& state, MazeAction action);

private:
    int mRowLength;
    int mColumnLength;
    std::vector<MazeCellType> mGrid;
    float mDefaultReward;
    float mMoveProb;
    MazeState mAgentState;
};

/*
 * MazeEnvironmentクラスの実装
 */

MazeEnvironment::MazeEnvironment(
    int rowLength, int columnLength,
    const std::vector<MazeCellType>& grid, float moveProb) :
    mRowLength(rowLength),
    mColumnLength(columnLength),
    mGrid(grid),
    mDefaultReward(-0.04f),
    mMoveProb(moveProb),
    mAgentState()
{
    this->Reset();
}

const std::vector<MazeAction>& MazeEnvironment::AvailableActions() const
{
    static const std::vector<MazeAction> availableActions = {
        MazeAction::Up, MazeAction::Down,
        MazeAction::Left, MazeAction::Right
    };

    return availableActions;
}

std::vector<MazeState> MazeEnvironment::AvailableStates() const
{
    std::vector<MazeState> availableStates;

    for (int i = 0; i < this->mRowLength; ++i)
        for (int j = 0; j < this->mColumnLength; ++j)
            if (this->At(i, j) != MazeCellType::Block)
                availableStates.emplace_back(i, j);

    return availableStates;
}

bool MazeEnvironment::CanActionAt(const MazeState& state) const
{
    return this->At(state.Row(), state.Column()) == MazeCellType::Normal;
}

MazeCellType MazeEnvironment::At(int row, int col) const
{
    assert(0 <= row && row < this->mRowLength);
    assert(0 <= col && col < this->mColumnLength);

    return this->mGrid[this->mColumnLength * row + col];
}

std::map<MazeState, float> MazeEnvironment::Transition(
        const MazeState& state, MazeAction action)
{
    std::map<MazeState, float> transitionProbs;

    /* 既に終端状態に達している場合 */
    if (!this->CanActionAt(state))
        return transitionProbs;

    MazeAction oppositeDirection = static_cast<MazeAction>(static_cast<int>(action) * -1);

    for (auto a : this->AvailableActions()) {
        float prob =
            (a == action) ? this->mMoveProb :
            (a != oppositeDirection) ? (1.0f - this->mMoveProb) / 2.0f : 0.0f;

        MazeState nextState = this->Move(state, a);
        transitionProbs[nextState] += prob;

        /*
        decltype(transitionProbs)::iterator it = transitionProbs.find(nextState);

        if (it != transitionProbs.end())
            transitionProbs[nextState] = prob;
        else
            transitionProbs[nextState] += prob;
        */
    }

    return transitionProbs;
}

std::tuple<float, bool> MazeEnvironment::Reward(const MazeState& state)
{
    float reward = this->mDefaultReward;
    bool done = false;
    
    MazeCellType cellType = this->At(state.Row(), state.Column());

    switch (cellType) {
        case MazeCellType::Reward:
            reward = 1.0f;
            done = true;
            break;
        case MazeCellType::Damage:
            reward = -1.0f;
            done = true;
            break;
    }

    return std::make_tuple(reward, done);
}

MazeState MazeEnvironment::Move(const MazeState& state, MazeAction action)
{
    if (!this->CanActionAt(state))
        throw std::runtime_error("Cannot move from here");
    
    int row = state.Row();
    int column = state.Column();

    switch (action) {
        case MazeAction::Up:
            row -= 1;
            break;
        case MazeAction::Down:
            row += 1;
            break;
        case MazeAction::Left:
            column -= 1;
            break;
        case MazeAction::Right:
            column += 1;
            break;
    }

    /* 迷路の外に出る場合は移動できない */
    if ((row < 0) || (this->mRowLength <= row))
        row = state.Row();
    
    if ((column < 0) || (this->mColumnLength <= column))
        column = state.Column();
    
    /* ブロックに衝突する場合は移動できない */
    if (this->At(row, column) == MazeCellType::Block) {
        row = state.Row();
        column = state.Column();
    }

    return MazeState(row, column);
}

MazeState MazeEnvironment::Reset()
{
    /* エージェントの位置を左下に初期化 */
    this->mAgentState = MazeState(this->mRowLength - 1, 0);

    return this->mAgentState;
}

std::tuple<std::optional<MazeState>, std::optional<float>, bool>
MazeEnvironment::Step(MazeAction action)
{
    std::optional<MazeState> nextState;
    std::optional<float> reward;
    bool done;

    std::tie(nextState, reward, done) = this->Transit(this->mAgentState, action);

    if (nextState)
        this->mAgentState = nextState.value();

    return std::make_tuple(nextState, reward, done);
}

std::tuple<std::optional<MazeState>, std::optional<float>, bool>
MazeEnvironment::Transit(
    const MazeState& state, MazeAction action)
{
    std::map<MazeState, float> transitionProbs = this->Transition(state, action);

    if (transitionProbs.empty())
        return std::make_tuple(std::nullopt, std::nullopt, true);
    
    std::vector<MazeState> nextStates;
    std::vector<float> nextProbs;

    for (const auto& stateProb : transitionProbs) {
        nextStates.push_back(stateProb.first);
        nextProbs.push_back(stateProb.second);
    }

    MazeState nextState = nextStates[RandomChoice(nextProbs)];

    float reward;
    bool done;
    std::tie(reward, done) = this->Reward(nextState);

    return std::make_tuple(nextState, reward, done);
}

#endif /* MAZE_HPP */
