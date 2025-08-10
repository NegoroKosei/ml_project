import numpy as np
import common.gridworld_render as render_helper

class GridWorld:
    def __init__(self):
        self.action_space = [0, 1, 2, 3]  # 上、下、左、右
        self.action_meaning = {
            0: "UP",
            1: "DOWN",
            2: "LEFT",
            3: "RIGHT"
        }

        self.reward_map = np.array(
            [[0, 0, 0, 1.0],
             [0, None, 0, -1.0],
             [0, 0, 0, 0]]
        )
        self.goal_state = (0, 3) # ゴールの位置
        self.wall_state = (1, 1) # 壁の位置
        self.start_state = (2, 0) # 開始位置
        self.agent_state = self.start_state # エージェントの位置

    # property デコレータを使用して、属性(変数)のようにアクセスできるようにする = ()なしで値を取得できる
    @property
    def height(self):
        return len(self.reward_map)
    @property
    def width(self):
        return len(self.reward_map[0])
    @property
    def shape(self):
        return self.reward_map.shape
    def actions(self):
        return self.action_space # [0, 1, 2, 3]
    
    def states(self):
        for h in range(self.height):
            for w in range(self.width):
                # yieldはジェネレータを作成するためのキーワード(returnの代わりに使用)
                # ジェネレータは、値を一つずつ返すイテレータ
                # 関数の実行が一時停止するだけで、実行位置、ローカル変数の状態を保持する
                yield (h, w)

    def next_state(self, state, action):
        # ①移動先の座標を計算
        action_move_map = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        move = action_move_map[action]
        next_state = (state[0] + move[0], state[1] + move[1])
        ny, nx = next_state

        # ②移動先が壁か範囲外ならば、元の状態を返す
        if (ny, nx) == self.wall_state or not (0 <= ny < self.height and 0 <= nx < self.width):
            return state
        return next_state
    
    def reward(self, state, action, next_state):
        # 今回は報酬は次の状態にのみ依存する
        return self.reward_map[next_state]
    

    def reset(self):
        self.agent_state = self.start_state
        return self.agent_state

    def step(self, action):
        state = self.agent_state
        next_state = self.next_state(state, action)
        reward = self.reward(state, action, next_state)
        done = (next_state == self.goal_state)

        self.agent_state = next_state
        return next_state, reward, done

    def render_v(self, v=None, policy=None, print_value=True):
        renderer = render_helper.Renderer(self.reward_map, self.goal_state,
                                          self.wall_state)
        renderer.render_v(v, policy, print_value)

    def render_q(self, q=None, print_value=True):
        renderer = render_helper.Renderer(self.reward_map, self.goal_state,
                                          self.wall_state)
        renderer.render_q(q, print_value)
