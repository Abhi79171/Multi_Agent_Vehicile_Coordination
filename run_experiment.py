import pygame
import sys
import time
import argparse
import openai
import random
from gpt4_vision_outputs import gpt_4_vision_outputs
from validation import discrete_distill, failure

pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Autonomous Vehicle Coordination Simulation")
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (150, 150, 150)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 100, 255)
LANE_WIDTH = WIDTH // 2
CAR_WIDTH, CAR_HEIGHT = 50, 80

class Car:
    def __init__(self, start_lane, y, color, label):
        self.original_lane = start_lane
        self.x = 100 if start_lane == "left" else 500
        self.target_x = self.x
        self.y = y
        self.color = color
        self.label = label
        self.speed = 2
        self.lane = start_lane
        self.switching = False

    def draw(self):
        pygame.draw.rect(screen, self.color, (self.x, self.y, CAR_WIDTH, CAR_HEIGHT))
        font = pygame.font.SysFont(None, 20)
        label_surface = font.render(self.label, True, BLACK)
        screen.blit(label_surface, (self.x + 5, self.y + 30))

    def update_position(self, action):
        if action == "left" and self.lane == "right":
            self.target_x = 100
            self.lane = "left"
            self.switching = True
        elif action == "right" and self.lane == "left":
            self.target_x = 500
            self.lane = "right"
            self.switching = True
        elif action == "slow down":
            self.speed = 1
        else:
            self.speed = 2

    def move(self):
        self.y -= self.speed
        if self.switching:
            if self.x < self.target_x:
                self.x += 5
                if self.x >= self.target_x:
                    self.x = self.target_x
                    self.switching = False
            elif self.x > self.target_x:
                self.x -= 5
                if self.x <= self.target_x:
                    self.x = self.target_x
                    self.switching = False

    def rect(self):
        return pygame.Rect(self.x, self.y, CAR_WIDTH, CAR_HEIGHT)


def run_pygame_iteration(model_name, car1_lane, car2_lane, amb_lane, car1_action, car2_action, ambulance_side, result):
    car1 = Car(car1_lane, HEIGHT - 100, BLUE, f"Car 1 ({model_name})")
    car2 = Car(car2_lane, HEIGHT - 200, GREEN, f"Car 2 ({model_name})")
    ambulance = Car(amb_lane, -100, RED, "Ambulance")
    ambulance.speed = 4

    car1.update_position(car1_action)
    car2.update_position(car2_action)

    font = pygame.font.SysFont(None, 28)
    font_small = pygame.font.SysFont(None, 22)
    timer = 0
    collision = False

    running = True
    while running:
        screen.fill(GRAY)
        pygame.draw.line(screen, WHITE, (LANE_WIDTH, 0), (LANE_WIDTH, HEIGHT), 5)

        # Lane labels
        left_label = font_small.render("LEFT LANE", True, WHITE)
        right_label = font_small.render("RIGHT LANE", True, WHITE)
        screen.blit(left_label, (100, 10))
        screen.blit(right_label, (500, 10))

        car1.draw()
        car2.draw()
        ambulance.draw()

        car1.move()
        car2.move()
        ambulance.y += ambulance.speed

        if car1.rect().colliderect(ambulance.rect()) or car2.rect().colliderect(ambulance.rect()):
            collision = True

        summary1 = f"Car 1: {car1.original_lane} → {car1.lane}"
        summary2 = f"Car 2: {car2.original_lane} → {car2.lane}"
        amb_summary = f"Ambulance lane: {ambulance_side}"
        screen.blit(font_small.render(summary1, True, BLACK), (10, HEIGHT - 60))
        screen.blit(font_small.render(summary2, True, BLACK), (10, HEIGHT - 40))
        screen.blit(font_small.render(amb_summary, True, BLACK), (10, HEIGHT - 20))

        final_result = result and not collision
        status_text = font.render(f"{model_name} - {'SUCCESS' if final_result else ' FAILURE'}", True, GREEN if final_result else RED)
        screen.blit(status_text, (10, 50))

        pygame.display.flip()
        pygame.time.delay(50)
        timer += 1

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN or timer > 150:
                running = False
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()


def experiment(usa_model, india_model, iterations):
    ambulance_side_center, ambulance_side_right, ambulance_side_left = gpt_4_vision_outputs()
    fms_finetuned_with_comm = 0
    fms_finetuned_no_comm = 0
    fms_base = 0

    for i in range(iterations):
        print(f"Current iteration: {i}")
        ambulance_side = random.choice(["right", "left"])
        car1_side = ambulance_side if random.random() < 0.8 else ("left" if ambulance_side == "right" else "right")
        car2_side = ambulance_side if random.random() < 0.8 else ("left" if ambulance_side == "right" else "right")

        model_input1 = f"Emergency vehicle approaching from {ambulance_side}. You are in a two-lane road, other car is in the {car2_side} lane."
        model_input2 = f"Emergency vehicle approaching from {ambulance_side}. You are in a two-lane road, other car is in the {car1_side} lane."

        content1 = f"You are on the {car1_side}. {model_input1} Output your action and a message about your intention."
        content2 = f"You are on the {car2_side}. {model_input2} Output your action and a message about your intention."

        response_usa = openai.ChatCompletion.create(model=usa_model, messages=[{"role": "system", "content": "..."}, {"role": "user", "content": content1}])
        usa_output = response_usa['choices'][0]['message']['content'].split('|')
        usa_action = usa_output[0].strip()
        usa_message = usa_output[1].strip() if len(usa_output) > 1 else "No message"

        response_india = openai.ChatCompletion.create(model=india_model, messages=[{"role": "system", "content": "..."}, {"role": "user", "content": content2}])
        india_output = response_india['choices'][0]['message']['content'].split('|')
        india_action = india_output[0].strip()
        india_message = india_output[1].strip() if len(india_output) > 1 else "No message"

        usa_adjusted = adjust_action(usa_action, india_message, ambulance_side, car1_side)
        india_adjusted = adjust_action(india_action, usa_message, ambulance_side, car2_side)
        usa_comm = discrete_distill(usa_adjusted)
        india_comm = discrete_distill(india_adjusted)
        fail_comm = failure(usa_comm, india_comm, ambulance_side, car1_side, car2_side)
        if fail_comm: fms_finetuned_with_comm += 1
        run_pygame_iteration("With Comm", car1_side, car2_side, ambulance_side, usa_comm, india_comm, ambulance_side, not fail_comm)

        usa_no_comm = discrete_distill(usa_action)
        india_no_comm = discrete_distill(india_action)
        fail_no_comm = failure(usa_no_comm, india_no_comm, ambulance_side, car1_side, car2_side)
        if fail_no_comm: fms_finetuned_no_comm += 1
        run_pygame_iteration("No Comm", car1_side, car2_side, ambulance_side, usa_no_comm, india_no_comm, ambulance_side, not fail_no_comm)

        response_base1 = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "system", "content": "..."}, {"role": "user", "content": content1}])
        base1 = discrete_distill(response_base1['choices'][0]['message']['content'])
        if base1 == "straight": base1 = car1_side

        response_base2 = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "system", "content": "..."}, {"role": "user", "content": content2}])
        base2 = discrete_distill(response_base2['choices'][0]['message']['content'])
        if base2 == "straight": base2 = car2_side

        fail_base = failure(base1, base2, ambulance_side, car1_side, car2_side)
        if fail_base: fms_base += 1
        run_pygame_iteration("Base", car1_side, car2_side, ambulance_side, base1, base2, ambulance_side, not fail_base)

    return fms_finetuned_with_comm, fms_finetuned_no_comm, fms_base

def adjust_action(action, received_message, ambulance_side, car_side):
    msg = received_message.lower()
    target_side = "left" if ambulance_side == "right" else "right"
    if car_side == ambulance_side:
        return target_side
    if target_side in msg:
        return target_side
    if ambulance_side in msg:
        return "slow down"
    if "slow down" in msg or "maintain" in msg:
        return target_side
    return target_side

def main():
    parser = argparse.ArgumentParser(description="Run the fine-tuning experiment with simulation")
    parser.add_argument("usa_model", help="Fine-tuned USA model name")
    parser.add_argument("india_model", help="Fine-tuned India model name")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations")
    args = parser.parse_args()
    fms_comm, fms_no_comm, fms_base = experiment(args.usa_model, args.india_model, args.iterations)
    print(f"\nSummary:\nFine-tuned model failures (with communication): {fms_comm}")
    print(f"Fine-tuned model failures (no communication): {fms_no_comm}")
    print(f"Base model failures: {fms_base}")

if __name__ == "__main__":
    main()
