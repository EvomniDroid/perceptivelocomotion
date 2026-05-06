import math

class FusionCostCalculator:
    def __init__(self, fall_rate_penalty: float = 1.0, urgency_reference_dist: float = 3.0):
        self.fall_rate_penalty = fall_rate_penalty
        self.urgency_reference_dist = urgency_reference_dist
        self.danger_ahead_threshold = 0.5

    def calculate_urgency(self, target_distance: float) -> float:
        if target_distance <= 0:
            return 1.0
        if target_distance > self.urgency_reference_dist * 3:
            return 0.0
        return 1.0 / (1.0 + target_distance / self.urgency_reference_dist)

    def predict_danger_ahead(self, robot_x: float, robot_y: float,
                            goal_x: float, goal_y: float,
                            get_fall_rate_func, 
                            check_distances: list = None) -> tuple:
        """
        预测前方是否有高风险区
        
        Args:
            robot_x, robot_y: 机器人当前位置
            goal_x, goal_y: 目标位置
            get_fall_rate_func: 获取某点摔倒率的函数,签名 get_fall_rate_func(x, y)
            check_distances: 要检查的距离列表,默认 [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
        
        Returns:
            (has_danger, first_danger_dist, max_danger_along_path)
            - has_danger: 前方是否有高风险区
            - first_danger_dist: 第一个高风险点距离
            - max_danger_along_path: 沿途最大危险度
        """
        if check_distances is None:
            check_distances = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]

        dx = goal_x - robot_x
        dy = goal_y - robot_y
        dist_to_goal = math.sqrt(dx * dx + dy * dy)
        
        if dist_to_goal < 0.1:
            return False, 0.0, 0.0
        
        dir_x = dx / dist_to_goal
        dir_y = dy / dist_to_goal
        
        has_danger = False
        first_danger_dist = 0.0
        max_danger = 0.0
        
        for d in check_distances:
            check_x = robot_x + dir_x * d
            check_y = robot_y + dir_y * d
            fr = get_fall_rate_func(check_x, check_y)
            
            max_danger = max(max_danger, fr)
            
            if fr > self.danger_ahead_threshold and first_danger_dist == 0.0:
                has_danger = True
                first_danger_dist = d
        
        return has_danger, first_danger_dist, max_danger

    def calculate_direction_penalty(self, frontier_x: float, frontier_y: float,
                                    robot_x: float, robot_y: float,
                                    goal_x: float, goal_y: float,
                                    has_danger_ahead: bool,
                                    danger_dist: float) -> float:
        """
        计算前沿点的方向惩罚因子
        如果预测前方有险,降低正前方方向得分,提高侧向绕路方向得分
        
        Returns:
            direction_penalty: 0.0~1.0,越低表示越应该绕路
        """
        dx = frontier_x - robot_x
        dy = frontier_y - robot_y
        dist_to_frontier = math.sqrt(dx * dx + dy * dy)
        
        if dist_to_frontier < 0.1:
            return 0.5
        
        front_dir_x = dx / dist_to_frontier
        front_dir_y = dy / dist_to_frontier
        
        to_goal_dx = goal_x - robot_x
        to_goal_dy = goal_y - robot_y
        goal_dist = math.sqrt(to_goal_dx * to_goal_dx + to_goal_dy * to_goal_dy)
        
        if goal_dist < 0.1:
            return 1.0
        
        goal_dir_x = to_goal_dx / goal_dist
        goal_dir_y = to_goal_dy / goal_dist
        
        dot = front_dir_x * goal_dir_x + front_dir_y * goal_dir_y
        
        if has_danger_ahead and danger_dist > 0:
            cross = front_dir_x * goal_dir_y - front_dir_y * goal_dir_x

            if cross > 0.3:
                绕路_bonus = min(0.5, (danger_dist - 1.0) * 0.15)
                return min(1.0, 1.0 + 绕路_bonus)
            elif cross < -0.3:
                绕路_bonus = min(0.5, (danger_dist - 1.0) * 0.15)
                return min(1.0, 1.0 + 绕路_bonus)
            elif dot > 0.7:
                前方危险_penalty = (danger_dist - 1.0) * 0.25
                return max(0.1, dot - 前方危险_penalty)
        
        return max(0.1, min(1.0, dot))

    def calculate_fusion_cost(self, point_x: float, point_y: float,
                               fall_rate: float, distance: float,
                               target_distance: float) -> float:
        urgency = self.calculate_urgency(target_distance)
        fall_weight = self.fall_rate_penalty * urgency
        distance_cost = distance
        fall_cost = fall_weight * fall_rate * 10.0
        return distance_cost + fall_cost

    def calculate_frontier_score(self, frontier: dict, target_distance: float,
                                  direction_score: float = 0.5,
                                  direction_penalty: float = 1.0) -> float:
        urgency = self.calculate_urgency(target_distance)
        fall_weight = self.fall_rate_penalty * urgency

        conf_val = max(0.1, min(2.0, float(frontier.get('conf', 0.5))))
        dist_val = max(0.3, float(frontier.get('dist', 1.0)))
        dist_score = min(1.0, 3.0 / dist_val) if dist_val > 0 else 0

        fall_rate = max(0.0, min(1.0, frontier.get('fall_rate', 0.0)))
        fall_factor = 1.0 - fall_weight * fall_rate

        score = conf_val * direction_score * direction_score * fall_factor * dist_score * direction_penalty
        return max(0.0, score)

def get_fusion_weight_description(urgency: float, fall_rate_penalty: float) -> str:
    fall_weight = fall_rate_penalty * urgency
    if fall_weight < 0.2:
        return "地形不考虑，目标优先"
    elif fall_weight < 0.5:
        return "地形轻微考虑"
    elif fall_weight < 1.0:
        return "地形中等考虑"
    else:
        return "地形优先，谨慎行动"
