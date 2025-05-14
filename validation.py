import openai


def discrete_distill(model_outputs):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": "Based on the action description, determine the car's action: 'right', 'left', 'straight', or 'slow down'. If the action involves slowing or stopping without a clear directional move, output 'slow down'. Output only one of these words."},
            {"role": "user", "content": model_outputs},
        ]
    )
    return response['choices'][0]['message']['content'].strip().lower()


def failure(car1, car2, ambulance, car1_side, car2_side):
    target_side = "left" if ambulance == "right" else "right"

    car1_blocks = car1 == ambulance and car1_side == ambulance
    car2_blocks = car2 == ambulance and car2_side == ambulance

    car1_clears = car1 == target_side
    car2_clears = car2 == target_side

    if car1_side == ambulance and car2_side == ambulance:
        return not (car1_clears and car2_clears)

    return (car1_blocks and not car2_clears) or (car2_blocks and not car1_clears)